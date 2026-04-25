#!/usr/bin/env python3
"""Profile the placement simulator to identify bottlenecks."""

import sys
import time
from pathlib import Path
import cProfile
import pstats
from io import StringIO

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from src.envs.placement_core import (
    empty_board,
    enumerate_candidates,
    extract_features,
    apply_placement,
    column_heights,
    count_holes,
    hole_depth,
    cumulative_wells,
)
from src.envs.placement_env import PlacementTetrisEnv

def profile_enumerate_candidates():
    """Profile enumerate_candidates on various board states."""
    print("\n" + "="*70)
    print("PROFILING: enumerate_candidates()")
    print("="*70)
    
    # Empty board
    board = empty_board()
    pieces = ["I", "O", "T", "J", "L", "S", "Z"]
    
    pr = cProfile.Profile()
    
    start = time.time()
    pr.enable()
    for _ in range(100):
        for piece in pieces:
            mask, candidates, outcomes = enumerate_candidates(board, piece)
    pr.disable()
    elapsed = time.time() - start
    
    s = StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
    ps.print_stats(20)
    print(s.getvalue())
    print(f"\n100 rounds x 7 pieces = 700 calls in {elapsed:.3f}s -> {700/elapsed:.0f} calls/sec")

def profile_feature_extraction():
    """Profile feature extraction functions."""
    print("\n" + "="*70)
    print("PROFILING: Feature extraction (column_heights, holes, etc)")
    print("="*70)
    
    board = empty_board()
    # Add some filled cells
    for y in range(5, 15):
        for x in range(3, 8):
            if (x + y) % 2 == 0:
                board[y, x] = 1
    
    pr = cProfile.Profile()
    pr.enable()
    start = time.time()
    for _ in range(1000):
        heights = column_heights(board)
        holes = count_holes(board)
        depth = hole_depth(board)
        wells = cumulative_wells(board)
    pr.disable()
    elapsed = time.time() - start
    
    s = StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
    ps.print_stats(20)
    print(s.getvalue())
    print(f"\n1000 feature extractions in {elapsed:.3f}s -> {1000/elapsed:.0f} extractions/sec")

def profile_full_rollout():
    """Profile a typical evaluation loop (like GA evaluator)."""
    print("\n" + "="*70)
    print("PROFILING: Full rollout (GA-style evaluation)")
    print("="*70)
    
    env = PlacementTetrisEnv(max_steps=500)
    
    pr = cProfile.Profile()
    
    start = time.time()
    pr.enable()
    
    for episode in range(20):
        obs, info = env.reset(seed=42 + episode)
        terminated = False
        truncated = False
        steps = 0
        while not (terminated or truncated) and steps < 100:
            # Simulate the GA evaluator hot path
            valid_actions = np.flatnonzero(env.action_masks())
            if valid_actions.size == 0:
                break
            action = int(valid_actions[0])
            obs, reward, terminated, truncated, info = env.step(action)
            steps += 1
    
    pr.disable()
    elapsed = time.time() - start
    
    s = StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
    ps.print_stats(20)
    print(s.getvalue())
    print(f"\n20 episodes in {elapsed:.3f}s")

if __name__ == "__main__":
    profile_enumerate_candidates()
    profile_feature_extraction()
    profile_full_rollout()
    print("\n" + "="*70)
    print("Profiling complete")
    print("="*70)
