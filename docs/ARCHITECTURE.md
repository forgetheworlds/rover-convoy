# Codebase guide

This project compares a learned centralized convoy controller with a reconstructed
leader/follower MPC on the **same coupled MuJoCo dynamics**. Each controller
receives route and state estimates; it returns high-level commands every 100 ms.

## Module map

| File | Responsibility |
| --- | --- |
| `src/data.py` | Frozen `Scenario`, human-readable normalized `ConvoyAction`, stable observation schema and feature names |
| `src/config.py` | Physical constants, speed sets, scenario definitions, XML location |
| `src/routes.py` | Taught route, heading/curvature estimates, geometrical reference interpolation |
| `src/environment.py` | MuJoCo physics stepping, observations, actions, rewards and safety termination |
| `src/control.py` | Geometric reference controller for supervised warm starting |
| `src/training.py` | Actor imitation, transfer across input sizes, PPO training |
| `src/evaluation.py` | Geometric-vs-PPO evaluation and CSV metrics |
| `src/convoy.py` | Training/evaluation command-line interface |
| `src/benchmark.py` | Upstream D-MPC adapter and controller comparisons |
| `src/render_convoy.py` | Offline video rendering and overlays |
| `assets/convoy.xml` | Coupled rover physical plant: 12 wheels, front yaw/pitch mount, rear sliding ball joint |

The import chain is `data → config → routes → environment → training/evaluation`.
`benchmark.py` and `render_convoy.py` are separate consumers of the same
environment. Controllers never modify MuJoCo directly.

## Ten-hertz controller loop

1. `env.reset(seed)` samples a scenario, sets cargo/terrain, loads a route and returns an observation.
2. `env._observe()` builds the normalized flat feature array from the rover estimates, route previews and slider.
3. `policy.predict(obs)`, geometric control, **or** the leader/follower MPC returns a four-element action array.
4. `env.step(action)` limits requested speed/steering to the physical command and slew bounds, maps them to 12 wheel targets and 4 front steering actuators, and steps MuJoCo 10 times at 0.01 s each.
5. The environment scores progress, slider displacement, path errors, cargo tilt and control changes, and terminates on completion or safety failure.

### Policy action contract

The 4 entries are `[leader_speed, leader_steer, follower_speed, follower_steer]`.
They are **normalized commands**, not direct wheel speeds or forces.
Use `ConvoyAction.from_array(values)` to inspect them by name outside the hot loop.

### Observation contract and checkpoint compatibility

The feature order is documented precisely by `data.observation_names(kind)`.
Each rover contributes 9 route/motion features; 10 shared measurements follow.
The models add features in order; earlier columns are unchanged:

- **28**: original single-speed controller.
- **29**: above + requested speed ceiling.
- **35**: above + two future heading previews and pitch for each rover (existing delivery checkpoint).
- **41**: above + three future *signed curvature observations* for each rover (new adaptive-speed training mode).

The last six curvature features are normalized by the wheelbase. Their lookahead
distances scale with the **scenario's speed ceiling**, giving the model early
warning of curves. They **do not set the vehicle speed**. Old 35-input models
still load unchanged; `--adaptive-speed --initialize-from assets/policy.zip`
copies the old actor's existing input weights and initializes the six new
input columns to zero, while training a new critic.

## Learned speed adaptation

There is **no external curvature-to-speed envelope** in the new benchmark,
rendering, or environment. The route curvature is input information only.
For new policies, repeated PPO rollouts must discover when to slow down from
the original delivery objective (progress/time versus instability, route
deviation and terminal failure). The `ADAPTIVE_SPEEDS` curriculum samples
3 m/s targets more frequently to expose the difficult cases.

This is a *training setup*, **not a validated claim** that the policy already
slows down correctly. The committed `assets/policy.zip` is the previous
35-input policy; its historical 3 m/s unpaced results include failures.

## Experimental caution

The benchmark calls the pinned upstream VT&R3 leader and follower solvers through
`src/vtr3_mpc.py`. The MuJoCo adapter supplies an acceleration-feasible route preview,
synthetic pose estimates and shared actuator limits. The first command respects the
100 ms simulation step. The field build and ROS/LiDAR pipeline remain unverified.
D-MPC uses the pose-based gap while PPO receives a direct slider observation.
Compare completion, failure reasons, slider RMSE, path deviation and compute time
on identical scenario seeds.

The goal of this refactor is readability and separation of responsibilities.
The controller's hot path keeps arrays; dataclasses are only used for static
scenario configuration or manual inspection. The 41-input mode does additional
curvature interpolation per step, so any training-throughput claim requires a
real benchmark; there is no claim of identical throughput.

## Basic verification

```bash
python -m unittest discover -s tests -v
python src/convoy.py evaluate --model assets/policy.zip --speed-sweep
python src/benchmark.py --speed 0.75
python src/profile_environment.py  # compare 35 vs. 41 feature step costs
```

## Controller variants

`ConvoyActors` composes separate leader and follower networks and a centralized
value network. Static feature lists define which observation channels enter each
actor. The actor modules emit two normalized commands each. Only the training
critic reads the complete observation. The follower gets the leader's measured
forward speed and yaw rate plus relative state and slider signals. The upstream
D-MPC instead passes a predicted pose sequence.

`ForecastSliderConvoy` adds an outward-motion training penalty using slider
position and velocity. The PPO and SAC candidates and the two-actor policy
train with the same reward. `experiment_suite.py` keeps selection on development
courses separate from the reserved comparison. The two-lap case uses a local
arclength hint for branch selection where the synthetic route crosses itself.
