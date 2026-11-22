# Learned Control vs. Classical MPC for a Cargo Rover Convoy

A simulation-based study inspired by Krawciw et al., [*Sharing the Load: Autonomous Multi-Rover Cargo Transport*](https://arxiv.org/abs/2510.18766v2), comparing a learned PPO controller against the original VT&R3 distributed MPC implementation in MuJoCo.

## Motivation

After reading the paper, I started thinking about some of the trade-offs between classical and learned control methods in robotics. One of the things that interested me was whether recent developments in agentic coding could reduce the engineering overhead associated with developing simulation environments, training policies and running experiments.

I wanted to explore how a relatively small neural network would perform as a controller when given the same observations available to the MPC, particularly under different payloads, terrain conditions and operating speeds.

I used AI coding agents to implement the simulation and experimental infrastructure while directing the experimental design and evaluating the results.

## Scope and limitations

All results are simulation-based and depend on the accuracy of the simulated dynamics and the controller configurations. The purpose of this project is not to establish that learned control is superior to MPC, but to explore its potential advantages and limitations under a specific set of conditions.

MPC offers advantages in interpretability, explicit constraint handling and adaptation to changing objectives, while learned controllers may offer lower runtime computational costs. Physical experiments would be necessary to evaluate how well these findings transfer to real robotic systems.

## Simulation demonstrations

**Forward crest** (475 kg payload, 7.8° crest — left: pose-only learned policy, right: upstream VT&R3 D-MPC):

![Forward crest side-by-side](media/forward.gif)

**Reverse crest** (role-swapped direction-of-travel frame):

![Reverse crest side-by-side](media/reverse.gif)

## Comparative evaluation

Both controllers operate using matched observations: delayed noisy rover poses `(x, y, yaw)`, forward speed, steering angle, taught route, and a 3.75 s leader rollout. The follower computes Euclidean spacing error and closing rate from the noisy 2D poses (`field_pair.pose_derived_spacing`). The learned controller uses two independent 2×64 `Tanh` actors, while the distributed MPC uses an online Ipopt solver.

| Suite | Task | Learned (`pose-only`) | Upstream VT&R3 D-MPC |
| --- | --- | --- | --- |
| **Field-inspired simulation** (10 Hz, 3 seeds) | Forward crest | 3/3, 5.28 cm [4.70, 6.02] | 3/3, 6.38 cm [5.95, 6.93] |
| **Field-inspired simulation** (10 Hz, 3 seeds) | Forward figure-eight | 3/3, 4.61 cm [4.42, 4.76] | 3/3, 8.39 cm [8.05, 8.77] |
| **Field-inspired simulation** (10 Hz, 3 seeds) | Reverse crest | 3/3, 6.09 cm [5.32, 6.85] | 0/3, 11.96 cm [11.21, 12.49] |
| **Field-inspired simulation** (10 Hz, 3 seeds) | Reverse figure-eight | 2/3, 6.54 cm [5.71, 7.24] | 0/3, 18.71 cm [18.03, 19.66] |
| **Docking** (10 Hz, 2 cases) | Stop, role swap, reverse dock | 2/2, peak 6.13 cm | 0/2, peak 32.08 cm |
| **Held-out** (`n = 20`, 10 Hz) | Unseen in-distribution crests | 20/20, 2.95 cm [2.26, 3.66] | 20/20, 3.81 cm [2.83, 4.78] |
| **Shifted OOD** (`n = 20`, 10 Hz) | Heavy load, low friction, high noise | 20/20, 6.08 cm [5.30, 7.02] | 19/20, 8.14 cm [7.04, 9.38] |
| **High-speed** (50 Hz) | 0.75 / 1.50 / 3.00 m/s caps | 4.46 / 5.31 / 6.93 cm (3/3) | 7.09 / 6.56 / 8.08 cm (3/3) |
| **Variable span** (10 Hz, 3 seeds) | 3.5 / 4.15 / 5.5 m cargo span | 7.30 / 5.28 / 5.91 cm (9/9) | 7.86 / 6.38 / 7.36 cm (9/9) |
| **Compute** (CPU p95) | Per-tick inference / solve time | 0.38–0.44 ms | 8.93–13.81 ms |

Tracking results report cargo slider RMSE (cm), with 95% bootstrap confidence intervals where shown. Docking reports peak cargo slider error (cm), and the compute row reports p95 CPU latency (ms). Completion counts are shown where available. Raw CSVs: [`field.csv`](results/field.csv) · [`docking.csv`](results/docking.csv) · [`heldout.csv`](results/heldout.csv) · [`shifted.csv`](results/shifted.csv) · [`speeds.csv`](results/speeds.csv) · [`spans.csv`](results/spans.csv).

## Quick start

Requires Python 3.11.

```sh
pip install -r requirements.txt
git init runs/vtr3
git -C runs/vtr3 fetch --depth 1 https://github.com/utiasASRL/vtr3.git 76433e3e1a275511ece5097d832534f6a9202da1
git -C runs/vtr3 checkout --detach FETCH_HEAD

python -m unittest discover -s tests -v
PYTHONPATH=src python src/evaluate.py --suite all --out-dir results
```
