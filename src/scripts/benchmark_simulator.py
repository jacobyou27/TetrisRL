from __future__ import annotations

import sys
import time
from statistics import mean
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from src.agents.heuristic_core import (
    HeuristicWeights,
    choose_best_action_no_lookahead,
    choose_best_action_with_lookahead,
)
from src.envs.placement_core import empty_board, enumerate_candidates
from src.envs.placement_env import PlacementTetrisEnv


def benchmark_enumerate(
    board: np.ndarray,
    piece: str,
    repeats: int = 500,
    warmup: int = 20,
) -> tuple[float, float]:
    """Benchmark enumerate_candidates on a fixed board/piece.

    Returns:
        avg_seconds_per_call, calls_per_second
    """
    for _ in range(warmup):
        enumerate_candidates(board, piece)

    t0 = time.perf_counter()
    for _ in range(repeats):
        enumerate_candidates(board, piece)
    total = time.perf_counter() - t0

    avg = total / repeats
    per_sec = repeats / total if total > 0 else float("inf")
    return avg, per_sec


def benchmark_rollout(
    steps: int = 300,
    repeats: int = 10,
    warmup: int = 1,
    lookahead_depth: int = 0,
    lookahead_weight: float = 0.6,
    seed_base: int = 0,
) -> tuple[float, float, float]:
    """Benchmark full rollout throughput.

    Measures reset + episode loop, but excludes env construction cost.
    Returns:
        avg_episode_seconds, steps_per_second, mean_steps_per_episode
    """
    weights = HeuristicWeights()
    env = PlacementTetrisEnv(max_steps=steps)

    try:
        # Warmup episodes
        for i in range(warmup):
            env.reset(seed=seed_base + i)
            terminated = False
            truncated = False
            while not (terminated or truncated):
                if not env.has_valid_actions():
                    break
                if lookahead_depth > 0:
                    action, _ = choose_best_action_with_lookahead(env, weights, lookahead_weight)
                else:
                    action, _ = choose_best_action_no_lookahead(env, weights)
                _, _, terminated, truncated, _ = env.step(action)

        episode_times: list[float] = []
        episode_steps: list[int] = []

        for i in range(repeats):
            t0 = time.perf_counter()
            env.reset(seed=seed_base + warmup + i)

            terminated = False
            truncated = False
            steps_this_episode = 0

            while not (terminated or truncated):
                if not env.has_valid_actions():
                    break

                if lookahead_depth > 0:
                    action, _ = choose_best_action_with_lookahead(env, weights, lookahead_weight)
                else:
                    action, _ = choose_best_action_no_lookahead(env, weights)

                _, _, terminated, truncated, _ = env.step(action)
                steps_this_episode += 1

            episode_times.append(time.perf_counter() - t0)
            episode_steps.append(steps_this_episode)

    finally:
        env.close()

    total_time = sum(episode_times)
    total_steps = sum(episode_steps)

    avg_episode_s = mean(episode_times) if episode_times else 0.0
    steps_per_sec = total_steps / total_time if total_time > 0 else 0.0
    mean_steps = mean(episode_steps) if episode_steps else 0.0
    return avg_episode_s, steps_per_sec, mean_steps


def main() -> None:
    empty = empty_board()

    cluttered = empty.copy()
    cluttered[19, :] = 1
    cluttered[18, ::2] = 1
    cluttered[17, 1::2] = 1
    cluttered[16, [2, 3, 6, 7]] = 1
    cluttered[15, [0, 9]] = 1

    print("Enumerate candidates throughput")
    for board_name, board in (("empty", empty), ("cluttered", cluttered)):
        for piece in ("I", "T", "L"):
            avg_s, per_sec = benchmark_enumerate(board, piece, repeats=500, warmup=20)
            print(
                f"  {board_name:9s} piece={piece} "
                f"avg={avg_s * 1000:.3f} ms  throughput={per_sec:.1f}/s"
            )

    avg_episode_s, steps_per_sec, mean_steps = benchmark_rollout(
        steps=300,
        repeats=10,
        warmup=1,
        lookahead_depth=0,
        seed_base=0,
    )
    print("\nRollout benchmark (no lookahead)")
    print(f"  avg episode time:     {avg_episode_s:.3f} s")
    print(f"  mean steps/episode:   {mean_steps:.1f}")
    print(f"  rollout steps/sec:    {steps_per_sec:.1f}")

    avg_episode_s_l1, steps_per_sec_l1, mean_steps_l1 = benchmark_rollout(
        steps=300,
        repeats=5,
        warmup=1,
        lookahead_depth=1,
        lookahead_weight=1.0,
        seed_base=1000,
    )
    print("\nRollout benchmark (lookahead=1.0)")
    print(f"  avg episode time:     {avg_episode_s_l1:.3f} s")
    print(f"  mean steps/episode:   {mean_steps_l1:.1f}")
    print(f"  rollout steps/sec:    {steps_per_sec_l1:.1f}")


if __name__ == "__main__":
    main()