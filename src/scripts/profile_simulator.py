#!/usr/bin/env python3
"""Profiler for the current placement-based Tetris project.

Profiles:
1. enumerate_candidates() throughput on representative boards
2. Heuristic no-lookahead rollout throughput
3. Heuristic lookahead rollout throughput
4. cProfile hot-path breakdowns for enumerate + rollout
"""

from __future__ import annotations

import sys
import time
import cProfile
import pstats
from io import StringIO
from pathlib import Path
from statistics import mean

PROJECT_ROOT = Path(__file__).resolve().parents[1]
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


def make_cluttered_board() -> np.ndarray:
    board = empty_board()
    board[19, :] = 1
    board[18, ::2] = 1
    board[17, 1::2] = 1
    board[16, [2, 3, 6, 7]] = 1
    board[15, [0, 9]] = 1
    board[14, [1, 4, 5, 8]] = 1
    return board


def print_profile(pr: cProfile.Profile, title: str, limit: int = 20) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)
    s = StringIO()
    stats = pstats.Stats(pr, stream=s).sort_stats("cumulative")
    stats.print_stats(limit)
    print(s.getvalue())


def bench_enumerate(
    board: np.ndarray,
    piece: str,
    repeats: int = 300,
    warmup: int = 20,
) -> tuple[float, float]:
    for _ in range(warmup):
        enumerate_candidates(board, piece)

    t0 = time.perf_counter()
    for _ in range(repeats):
        enumerate_candidates(board, piece)
    total = time.perf_counter() - t0

    avg_s = total / repeats
    per_sec = repeats / total if total > 0 else float("inf")
    return avg_s, per_sec


def profile_enumerate_candidates(repeats: int = 100) -> None:
    board = empty_board()
    pieces = ["I", "O", "T", "J", "L", "S", "Z"]

    pr = cProfile.Profile()
    t0 = time.perf_counter()
    pr.enable()
    for _ in range(repeats):
        for piece in pieces:
            enumerate_candidates(board, piece)
    pr.disable()
    elapsed = time.perf_counter() - t0

    print_profile(pr, "CPROFILE: enumerate_candidates()")
    total_calls = repeats * len(pieces)
    print(f"{repeats} rounds x {len(pieces)} pieces = {total_calls} calls in {elapsed:.3f}s")
    print(f"Overall throughput: {total_calls / elapsed:.1f} calls/sec")


def run_rollout_once(
    *,
    max_steps: int,
    seed: int,
    lookahead_depth: int,
    lookahead_weight: float,
    preset: str | None = None,
) -> tuple[float, int, bool]:
    weights = HeuristicWeights()
    env_kwargs = {"max_steps": max_steps}
    if preset is not None:
        env_kwargs["initial_board_preset"] = preset

    env = PlacementTetrisEnv(**env_kwargs)
    try:
        env.reset(seed=seed)
        terminated = False
        truncated = False
        steps = 0

        t0 = time.perf_counter()
        while not (terminated or truncated):
            if not env.has_valid_actions():
                break

            if lookahead_depth > 0:
                action, _ = choose_best_action_with_lookahead(env, weights, lookahead_weight)
            else:
                action, _ = choose_best_action_no_lookahead(env, weights)

            _, _, terminated, truncated, _ = env.step(action)
            steps += 1

        elapsed = time.perf_counter() - t0
        topout = bool(terminated and not truncated and env.step_count < env.max_steps)
        return elapsed, steps, topout
    finally:
        env.close()


def bench_rollout(
    *,
    repeats: int = 10,
    max_steps: int = 300,
    lookahead_depth: int = 0,
    lookahead_weight: float = 1.0,
    seed_base: int = 100,
    preset: str | None = None,
) -> tuple[float, float, float, float]:
    episode_times: list[float] = []
    episode_steps: list[int] = []
    topouts = 0

    # Warmup
    run_rollout_once(
        max_steps=max_steps,
        seed=seed_base - 1,
        lookahead_depth=lookahead_depth,
        lookahead_weight=lookahead_weight,
        preset=preset,
    )

    for i in range(repeats):
        elapsed, steps, topout = run_rollout_once(
            max_steps=max_steps,
            seed=seed_base + i,
            lookahead_depth=lookahead_depth,
            lookahead_weight=lookahead_weight,
            preset=preset,
        )
        episode_times.append(elapsed)
        episode_steps.append(steps)
        topouts += int(topout)

    total_time = sum(episode_times)
    total_steps = sum(episode_steps)
    avg_episode_s = mean(episode_times) if episode_times else 0.0
    steps_per_sec = total_steps / total_time if total_time > 0 else 0.0
    mean_steps = mean(episode_steps) if episode_steps else 0.0
    topout_rate = topouts / repeats if repeats > 0 else 0.0
    return avg_episode_s, steps_per_sec, mean_steps, topout_rate


def profile_rollout(
    *,
    repeats: int = 5,
    max_steps: int = 300,
    lookahead_depth: int = 0,
    lookahead_weight: float = 1.0,
    seed_base: int = 1000,
) -> None:
    weights = HeuristicWeights()
    env = PlacementTetrisEnv(max_steps=max_steps)

    pr = cProfile.Profile()
    t0 = time.perf_counter()
    pr.enable()

    for ep in range(repeats):
        env.reset(seed=seed_base + ep)
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

    pr.disable()
    elapsed = time.perf_counter() - t0
    env.close()

    label = f"CPROFILE: rollout (lookahead_depth={lookahead_depth}, lookahead_weight={lookahead_weight})"
    print_profile(pr, label)
    print(f"{repeats} episodes profiled in {elapsed:.3f}s")


def main() -> None:
    empty = empty_board()
    cluttered = make_cluttered_board()

    print("\nEnumerate candidates throughput")
    for board_name, board in (("empty", empty), ("cluttered", cluttered)):
        for piece in ("I", "T", "L"):
            avg_s, per_sec = bench_enumerate(board, piece, repeats=300, warmup=20)
            print(
                f"  {board_name:9s} piece={piece} "
                f"avg={avg_s * 1000:.3f} ms  throughput={per_sec:.1f}/s"
            )

    avg_ep_s, steps_per_sec, mean_steps, topout_rate = bench_rollout(
        repeats=10,
        max_steps=300,
        lookahead_depth=0,
        lookahead_weight=1.0,
        seed_base=200,
    )
    print("\nRollout benchmark (heuristic, no lookahead)")
    print(f"  avg episode time:   {avg_ep_s:.3f} s")
    print(f"  mean steps/episode: {mean_steps:.1f}")
    print(f"  rollout steps/sec:  {steps_per_sec:.1f}")
    print(f"  top-out rate:       {100 * topout_rate:.1f}%")

    avg_ep_s, steps_per_sec, mean_steps, topout_rate = bench_rollout(
        repeats=5,
        max_steps=300,
        lookahead_depth=1,
        lookahead_weight=1.0,
        seed_base=400,
    )
    print("\nRollout benchmark (heuristic, lookahead=1.0)")
    print(f"  avg episode time:   {avg_ep_s:.3f} s")
    print(f"  mean steps/episode: {mean_steps:.1f}")
    print(f"  rollout steps/sec:  {steps_per_sec:.1f}")
    print(f"  top-out rate:       {100 * topout_rate:.1f}%")

    profile_enumerate_candidates(repeats=100)
    profile_rollout(repeats=5, max_steps=300, lookahead_depth=0, lookahead_weight=1.0, seed_base=800)
    profile_rollout(repeats=3, max_steps=300, lookahead_depth=1, lookahead_weight=1.0, seed_base=1200)

    print("\n" + "=" * 78)
    print("Profiling complete")
    print("=" * 78)


if __name__ == "__main__":
    main()