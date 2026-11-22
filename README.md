# Learning to drive a cargo rover convoy

Two simulated lunar rovers carry a shared payload through a coupled MuJoCo
model. This project explores whether a centralized PPO policy can coordinate
both vehicles while maintaining path tracking and minimizing displacement of
the cargo slider. The starting point is *Sharing the Load*:
[paper](https://arxiv.org/abs/2510.18766v2) and the
[public VT&R3 controller code](https://github.com/utiasASRL/vtr3).

This is an independent simulation investigation, **not** a validated
replacement for the paper's field-tested distributed MPC.

## What the code does

```text
Taught path + pose estimates + rover motion + slider measurement
                             |
                    Policy observations
                             |
                PPO or upstream VT&R3 D-MPC
                             |
               Speed + steering for each rover
                             |
              Wheel targets and coupled MuJoCo
                             |
                  Rewards and benchmark data
```

- **Physical plant:** two six-wheel rovers, 475 kg nominal payload, yaw/pitch front
  connection and ±50 cm rear slider, terrain height field and tire friction.
  Geometry and tire/suspension behavior are approximations.
- **Learned controller:** one PPO network running at 10 Hz, outputting speed and
  steering for both rovers. Training begins with imitation of a geometric
  controller and proceeds through speed-conditioned and delivery stages.
- **MPC baseline:** the pinned upstream VT&R3 CasADi/IPOPT leader and follower
  solvers, called through a MuJoCo adapter. The leader's predicted poses inform the follower's optimization.
- **Inputs:** synthetic route-localization errors, rover motion, slider readings
  and upcoming route information, **not** raw LiDAR or camera pixels.

### Watch a legacy checkpoint

**Flat speed test:** the committed 35-input delivery policy runs a 60 m
straight route with a 475 kg payload and a 3 m/s command ceiling, completing
the illustrated run in 25.1 simulated seconds.

![PPO flat route simulation](media/flat-speed.gif)

The older [loose-crest recording](media/loose-crest.gif) used a shared,
hand-designed curvature speed envelope. It is retained as a historical
artifact and **is not evidence that PPO learned to brake for bends**.
That external envelope has been removed from the current benchmark and
rendering paths.

## Current unpaced comparison

These are results reported for the original committed 35-input checkpoints
and the reconstructed D-MPC, before the code refactor. Each controller is
tested on five scenario/seed combinations per command ceiling. The
refactor does not establish new results for the 41-input policy.

| Speed ceiling | Reconstructed D-MPC complete | PPO seed 0 complete | PPO seed 1 complete |
| --- | ---: | ---: | ---: |
| 0.75 m/s | 2/5 | 5/5 | 5/5 |
| 1.50 m/s | 2/5 | 5/5 | 5/5 |
| 3.00 m/s | 4/5 | 2/5 | 2/5 |

At 3 m/s **without pacing**, both legacy PPO seeds can leave the allowed
path on challenging bends. Those failures motivate learned anticipation;
they are not hidden by an external speed cap. The reconstructed MPC also
needs validation against the original released controller before attributing
comparative improvements to learning.

## What changed: learn speed adaptation instead of applying an envelope

The new `--adaptive-speed` option changes the delivery-policy observation
from **35 to 41 features**, appending three upcoming signed path-curvature
samples for **each rover**. The curvature is information only. The policy
still chooses both speed commands itself, subject only to the original
physical speed/steering and actuator-rate limits.

Training samples high-speed targets more often and uses the same
delivery objective: finish quickly while avoiding slider excursions,
path deviation, cargo tilt and terminal failures. This gives PPO a chance
to learn to **slow before turns and accelerate afterward**.

**No adaptive-speed checkpoint is included yet.** These changes provide
the training/evaluation pathway, not a demonstrated improvement.

## Quick start

Python 3.11 is recommended. Install FFmpeg to export MP4 videos.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Smoke tests for schemas, route previews, MuJoCo stepping, and determinism
python -m unittest discover -s tests -v

# Check out the original solver sources at the benchmark's pinned revision.
git init runs/vtr3
git -C runs/vtr3 fetch --depth 1 https://github.com/utiasASRL/vtr3.git 76433e3e1a275511ece5097d832534f6a9202da1
git -C runs/vtr3 checkout --detach FETCH_HEAD

# Existing 35-feature checkpoint and upstream D-MPC
python src/benchmark.py --speed 0.75
python src/benchmark.py --speed 3
python src/render_convoy.py --model assets/policy.zip --speed-limit 3 --course flat --playback 2
```

The benchmark checks both solver files against the pinned source hashes and records
the VT&R3 revision in its CSV. Use `--vtr3 PATH` to point at an existing checkout
and `--course flat` to run a single-course smoke comparison. The historical results
above used the prior reconstructed baseline. Rerun the benchmark for results with
the upstream solvers.

### Train a policy that observes curvature and learns its own speed decisions

```bash
python src/convoy.py train --delivery --adaptive-speed --workers 4 \
    --steps 524288 --seed 1 --initialize-from assets/policy.zip \
    --out runs/learned-speed

# Compare old and new PPO models with the same upstream D-MPC
python src/benchmark.py --speed 3 \
    --model assets/policy.zip --model runs/learned-speed/ppo.zip

# Display the new checkpoint with no external speed scheduling
python src/render_convoy.py --model runs/learned-speed/ppo.zip \
    --speed-limit 3 --course loose-crest
```

The warm start copies all matching actor weights and sets the new
curvature-input weights to zero before PPO training. It uses a fresh critic.
The network architecture and its four output commands remain the same.

## Where things live

| File | Responsibility |
| --- | --- |
| [`src/data.py`](src/data.py) | Scenario/action dataclasses, positional observation feature names |
| [`src/config.py`](src/config.py) | Physical constants and training/evaluation scenario presets |
| [`src/routes.py`](src/routes.py) | Taught-route geometry, tracking, curvature and reference poses |
| [`src/environment.py`](src/environment.py) | MuJoCo physics, policy observations, actions, rewards, safety checks |
| [`src/control.py`](src/control.py) | Geometric imitation controller |
| [`src/training.py`](src/training.py) | PPO training and actor transfer |
| [`src/evaluation.py`](src/evaluation.py) | Geometric/PPO comparison and CSV output |
| [`src/convoy.py`](src/convoy.py) | Small train/evaluate CLI |
| [`src/benchmark.py`](src/benchmark.py) | Shared MuJoCo comparison for PPO and D-MPC |
| [`src/vtr3_mpc.py`](src/vtr3_mpc.py) | Source integrity checks and adapter for the upstream solvers |
| [`src/render_convoy.py`](src/render_convoy.py) | Visualization and MP4 export |
| [`assets/convoy.xml`](assets/convoy.xml) | Coupled MuJoCo rover model |

For the full data flow and exact input feature names, read
[the architecture guide](docs/ARCHITECTURE.md).

The simulator uses flat NumPy arrays in the time-critical control loop.
Dataclasses are used for configuration and human-readable inspection,
not for allocating objects on every 100 ms controller step.

## Limitations and next experiments

- The benchmark runs the published VT&R3 solver code. The simulator supplies route
  references, pose estimates and the common actuator limits. The field trial's exact
  revision, ROS/LiDAR pipeline and vehicle dynamics remain unverified here.
- PPO receives slider state and rover pitch in addition to the pose estimates used by
  D-MPC. The comparison measures the two complete controller designs with those inputs.
- Better high-speed completion must be demonstrated with **independent
  training seeds**, matching test scenarios, and raw CSV results.
- The new input previews add a small amount of per-step interpolation.
  Training throughput and controller latency should be measured, not assumed.
- These are simulated stress tests. The paper's loaded field-trial setting
  was a 0.5 m/s nominal speed with a 0.75 m/s command limit.
