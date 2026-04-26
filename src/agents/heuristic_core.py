
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

from src.envs.placement_core import enumerate_candidates

SAFE_MAX_HEIGHT = 10
WARN_MAX_HEIGHT = 14
CLEAN_HEIGHT_RELIEF = 0.5
DIRTY_HEIGHT_SCALE = 1.5
WELL_HEIGHT_RELIEF = 0.9


def max_height_penalty(features, weight: float) -> float:
    """Piecewise max-height penalty.

    We allow a clean mid-height stack to breathe, then ramp the penalty once the
    board gets meaningfully tall. Clean boards get relief; dirty boards get
    punished harder.
    """
    max_height = float(features.max_height)
    if max_height <= SAFE_MAX_HEIGHT:
        base = 0.0
    elif max_height <= WARN_MAX_HEIGHT:
        base = 0.5 * (max_height - SAFE_MAX_HEIGHT)
    else:
        excess = max_height - WARN_MAX_HEIGHT
        base = 2.0 + excess + 0.5 * excess * excess

    scale = 1.0
    holes = int(features.holes)
    rows_with_holes = int(features.rows_with_holes)
    wells = float(features.cumulative_wells)

    if holes == 0 and rows_with_holes == 0:
        scale *= CLEAN_HEIGHT_RELIEF
    elif holes >= 3 or rows_with_holes >= 2:
        scale *= DIRTY_HEIGHT_SCALE

    if wells > 0 and holes <= 1:
        scale *= WELL_HEIGHT_RELIEF

    return weight * base * scale



@dataclass(frozen=True)
class HeuristicWeights:
    # Tetris-friendly defaults for standalone evaluation.
    single_score: float = 4.0
    double_score: float = 16.0
    triple_score: float = 52.0
    tetris_score: float = 190.0
    aggregate_height_weight: float = 1.10
    holes_weight: float = 6.50
    bumpiness_weight: float = 0.18
    max_height_weight: float = 1.20
    row_transitions_weight: float = 1.10
    column_transitions_weight: float = 1.80
    rows_with_holes_weight: float = 1.60
    hole_depth_weight: float = 1.80
    cumulative_wells_weight: float = -0.35
    eroded_piece_cells_weight: float = 4.50


def board_value(features, weights: HeuristicWeights) -> float:
    return -(
        weights.aggregate_height_weight * float(features.aggregate_height)
        + weights.holes_weight * float(features.holes)
        + weights.bumpiness_weight * float(features.bumpiness)
        + max_height_penalty(features, weights.max_height_weight)
        + weights.row_transitions_weight * float(features.row_transitions)
        + weights.column_transitions_weight * float(features.column_transitions)
        + weights.rows_with_holes_weight * float(features.rows_with_holes)
        + weights.hole_depth_weight * float(features.hole_depth)
        + weights.cumulative_wells_weight * float(features.cumulative_wells)
    )


def score_delta_from_lines(lines: int, weights: HeuristicWeights) -> float:
    if lines == 1:
        return weights.single_score
    if lines == 2:
        return weights.double_score
    if lines == 3:
        return weights.triple_score
    if lines >= 4:
        return weights.tetris_score
    return 0.0


def score_outcome(outcome, weights: HeuristicWeights) -> float:
    return (
        score_delta_from_lines(int(outcome.lines_cleared), weights)
        + weights.eroded_piece_cells_weight * float(getattr(outcome, "eroded_piece_cells", 0))
        + board_value(outcome.features, weights)
    )


def _tie_tuple(outcome) -> tuple[int, int, int, int]:
    return (
        int(outcome.lines_cleared),
        -int(outcome.features.holes),
        -int(outcome.features.aggregate_height),
        -int(outcome.features.bumpiness),
    )


def _current_outcomes(env) -> Iterable:
    outcomes = getattr(env, "_outcomes", None)
    if outcomes:
        return outcomes
    _, _, outcomes = enumerate_candidates(env.board, env.current_piece)
    return outcomes


def choose_best_action_no_lookahead(env, weights: HeuristicWeights) -> tuple[int, float]:
    outcomes = _current_outcomes(env)
    if not outcomes:
        raise RuntimeError("No valid outcomes available.")

    best_action = None
    best_score = -float("inf")
    best_tiebreak = None

    for outcome in outcomes:
        score = score_outcome(outcome, weights)
        tie = _tie_tuple(outcome)
        if score > best_score or (math.isclose(score, best_score) and (best_tiebreak is None or tie > best_tiebreak)):
            best_score = score
            best_action = int(outcome.candidate.action_id)
            best_tiebreak = tie

    if best_action is None:
        raise RuntimeError("Failed to choose a best action.")
    return best_action, best_score


def choose_best_action_with_lookahead(env, weights: HeuristicWeights, lookahead_weight: float = 0.60) -> tuple[int, float]:
    first_outcomes = _current_outcomes(env)
    if not first_outcomes:
        raise RuntimeError("No valid outcomes available.")

    best_action = None
    best_total = -float("inf")
    best_tiebreak = None

    for first in first_outcomes:
        immediate = score_outcome(first, weights)
        _, _, second_outcomes = enumerate_candidates(first.board_after, env.next_piece)
        future = max((score_outcome(second, weights) for second in second_outcomes), default=-1e9)
        total = immediate + lookahead_weight * future
        tie = _tie_tuple(first)
        if total > best_total or (math.isclose(total, best_total) and (best_tiebreak is None or tie > best_tiebreak)):
            best_total = total
            best_action = int(first.candidate.action_id)
            best_tiebreak = tie

    if best_action is None:
        raise RuntimeError("Failed to choose a best action.")
    return best_action, best_total
