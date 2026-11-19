"""Measure the runtime overhead of curvature-preview observations.

Run: python src/profile_environment.py
This is a local descriptive benchmark, not a pass/fail performance test.
"""

from time import perf_counter_ns

import numpy as np

from config import Scenario
from environment import CargoConvoy


def measure(adaptive_speed, steps=400):
    env = CargoConvoy(Scenario("flat", distance=60.0, speed=3.0),
                      delivery=True, adaptive_speed=adaptive_speed)
    action = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    latencies_ms = []
    try:
        observation, _ = env.reset(seed=1000)
        for _ in range(steps):
            start = perf_counter_ns()
            observation, reward, ended, truncated, info = env.step(action)
            latencies_ms.append((perf_counter_ns() - start) / 1e6)
            if ended or truncated:
                observation, _ = env.reset(seed=1000)
    finally:
        env.close()
    return np.median(latencies_ms), np.percentile(latencies_ms, 95)


if __name__ == "__main__":
    legacy = measure(False)
    adaptive = measure(True)
    print(f"35 features: median {legacy[0]:.3f} ms, p95 {legacy[1]:.3f} ms")
    print(f"41 features: median {adaptive[0]:.3f} ms, p95 {adaptive[1]:.3f} ms")
    print("Includes MuJoCo stepping and Python observations, not PPO inference or training.")
