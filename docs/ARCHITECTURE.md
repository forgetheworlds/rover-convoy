# Controller and training

The simulator runs MuJoCo physics at 100 Hz and updates commands at 10 Hz.
The learned leader sees its own pose and route. The follower sees its own
state, slider measurements, and the leader's estimated pose sequence.
Each actor returns its own speed and steering.

The reverse adapter swaps the role order and expresses the route and pose
estimates in the direction-of-travel frame. It maps the outputs to signed
physical wheel and steering targets. The measured slider keeps its physical
meaning. During a role swap, the supervisor stops the pair, changes the
route and role order, and clears the delayed observation history.

- `environment.py`, `routes.py`, `config.py`, and `data.py` define the plant, route, and input layout.
- `learned_control.py`, `leader_preview.py`, and `field_pair.py` define the two actors, the forecast, and training.
- `vtr3_mpc.py` loads the pinned upstream solvers. `benchmark.py` runs either controller with the same plant.
- `direction_trials.py` and `maneuver.py` run the transport and docking comparisons. `render_convoy.py` exports replays.

## Reproduce the training

`assets/leader-follower.zip` contains the initialization weights. The
released checkpoints are included in `assets/field-forward.zip` and
`assets/field-bidirectional.zip`.

```sh
python src/field_pair.py --warm-from assets/leader-follower.zip --out runs/field-init
python src/field_pair.py --resume runs/field-init/policy.zip --slider-weight 0.50 --speed-weight 2.5 --out runs/field-spacing
python src/field_pair.py --adapt-from runs/field-spacing/rl_model_32768_steps.zip --mixed-direction --out runs/mixed-init
python src/field_pair.py --resume runs/mixed-init/policy.zip --mixed-direction --steps 32768 --slider-weight 0.50 --speed-weight 2.5 --out runs/mixed-ppo
python src/field_pair.py --adapt-from runs/mixed-ppo/policy.zip --mixed-direction --direction-input --seed 972 --out runs/direction-init
python src/field_pair.py --resume runs/direction-init/policy.zip --mixed-direction --direction-input --steps 16384 --slider-weight 0.50 --speed-weight 2.5 --seed 973 --out runs/bidirectional
```

## Reproduce the comparisons

Run the commands in the README for the trained pair. To repeat the
zero-shot reverse test, pass `assets/field-forward.zip` to
`direction_trials.py` with `--reverse`. For VT&R3, substitute
`--vtr3 runs/vtr3` for `--model`. Both direction comparisons use seeds 71, 91, and 111.

```sh
python -m unittest discover -s tests -v
python src/render_convoy.py --model assets/field-bidirectional.zip --from-seconds 58
python src/render_convoy.py --model assets/field-bidirectional.zip --reverse --from-seconds 58
```
