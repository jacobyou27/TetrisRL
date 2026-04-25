from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from .placement_core import (
    ACTION_SPACE_SIZE,
    BOARD_HEIGHT,
    BOARD_WIDTH,
    PieceId,
    PlacementFeatures,
    PlacementOutcome,
    board_to_ascii,
    clone_board,
    empty_board,
    enumerate_candidates,
    extract_features,
    make_preset_board,
)

PIECES: tuple[PieceId, ...] = ("I", "O", "T", "J", "L", "S", "Z")
PIECE_TO_INDEX = {piece: i for i, piece in enumerate(PIECES)}
INDEX_TO_PIECE = {i: piece for i, piece in enumerate(PIECES)}


@dataclass
class RewardConfig:
    single_score: float = 2.0
    double_score: float = 8.0
    triple_score: float = 32.0
    tetris_score: float = 120.0

    shaping_scale: float = 0.02
    shaping_gamma: float = 0.99

    aggregate_height_weight: float = 0.12
    holes_weight: float = 3.00
    bumpiness_weight: float = 0.08
    max_height_weight: float = 0.18
    row_transitions_weight: float = 0.50
    column_transitions_weight: float = 0.35
    rows_with_holes_weight: float = 1.00
    hole_depth_weight: float = 0.50
    cumulative_wells_weight: float = 0.60
    eroded_piece_cells_weight: float = 1.25

    survival_bonus: float = 0.0
    game_over_penalty: float = -25.0
    invalid_action_penalty: float = -5.0


SAFE_MAX_HEIGHT = 10
WARN_MAX_HEIGHT = 14
CLEAN_HEIGHT_RELIEF = 0.5
DIRTY_HEIGHT_SCALE = 1.5
WELL_HEIGHT_RELIEF = 0.9


def _max_height_penalty_from_features(feats: PlacementFeatures, weight: float) -> float:
    max_height = float(feats.max_height)
    if max_height <= SAFE_MAX_HEIGHT:
        base = 0.0
    elif max_height <= WARN_MAX_HEIGHT:
        base = 0.5 * (max_height - SAFE_MAX_HEIGHT)
    else:
        excess = max_height - WARN_MAX_HEIGHT
        base = 2.0 + excess + 0.5 * excess * excess

    scale = 1.0
    holes = int(feats.holes)
    rows_with_holes = int(feats.rows_with_holes)
    wells = float(feats.cumulative_wells)

    if holes == 0 and rows_with_holes == 0:
        scale *= CLEAN_HEIGHT_RELIEF
    elif holes >= 3 or rows_with_holes >= 2:
        scale *= DIRTY_HEIGHT_SCALE

    if wells > 0 and holes <= 1:
        scale *= WELL_HEIGHT_RELIEF

    return weight * base * scale


class PlacementTetrisEnv(gym.Env):
    metadata = {"render_modes": ["ansi", "rgb_array"], "render_fps": 4}

    def __init__(
        self,
        reward_config: RewardConfig | None = None,
        max_steps: int = 10_000,
        initial_board: np.ndarray | None = None,
        initial_board_preset: str | None = None,
        render_mode: str | None = None,
        render_upscale: int = 30,
    ) -> None:
        super().__init__()
        if render_mode is not None and render_mode not in self.metadata["render_modes"]:
            raise ValueError(f"Unsupported render_mode: {render_mode}")
        if initial_board is not None and initial_board_preset is not None:
            raise ValueError("Pass only one of initial_board or initial_board_preset.")
        if render_upscale <= 0:
            raise ValueError("render_upscale must be positive.")

        self.reward_config = reward_config or RewardConfig()
        self.max_steps = int(max_steps)
        self.render_mode = render_mode
        self.render_upscale = int(render_upscale)

        if initial_board_preset is not None:
            self._base_board = make_preset_board(initial_board_preset)
        elif initial_board is not None:
            self._base_board = clone_board(initial_board)
        else:
            self._base_board = empty_board()

        self.action_space = spaces.Discrete(ACTION_SPACE_SIZE)
        self.observation_space = spaces.Dict(
            {
                "board": spaces.Box(low=0, high=1, shape=(BOARD_HEIGHT, BOARD_WIDTH), dtype=np.uint8),
                "current_piece": spaces.Discrete(len(PIECES)),
                "next_piece": spaces.Discrete(len(PIECES)),
                "action_mask": spaces.Box(low=0, high=1, shape=(ACTION_SPACE_SIZE,), dtype=np.uint8),
            }
        )

        self.board: np.ndarray = empty_board()
        self.current_piece: PieceId = "I"
        self.next_piece: PieceId = "O"

        self.step_count = 0
        self.total_lines_cleared = 0
        self.total_score = 0.0

        self._bag: list[PieceId] = []
        self._mask = np.zeros(ACTION_SPACE_SIZE, dtype=bool)
        self._candidates = []
        self._outcomes = []
        self._outcome_by_action_id: dict[int, PlacementOutcome] = {}
        self._board_features: PlacementFeatures = extract_features(self.board)

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
        super().reset(seed=seed)
        options = options or {}
        self.step_count = 0
        self.total_lines_cleared = 0
        self.total_score = 0.0

        if "board" in options and "board_preset" in options:
            raise ValueError("Use only one of options['board'] or options['board_preset'].")

        if "board" in options:
            self.board = clone_board(options["board"])
        elif "board_preset" in options:
            self.board = make_preset_board(options["board_preset"])
        else:
            self.board = clone_board(self._base_board)

        self._bag = []
        self.current_piece = self._pop_piece()
        self.next_piece = self._pop_piece()
        self._refresh_candidates()

        obs = self._get_obs()
        info = self._get_info()
        info["score_delta"] = 0.0
        return obs, info

    def step(self, action: int) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        if not self.action_space.contains(action):
            raise ValueError(f"Action {action} is outside action space.")

        self.step_count += 1
        terminated = False
        truncated = False

        if not self._mask[action]:
            reward = float(self.reward_config.invalid_action_penalty)
            terminated = True
            info = self._get_info()
            info["invalid_action"] = True
            info["score_delta"] = 0.0
            return self._get_obs(), reward, terminated, truncated, info

        outcome = self._outcome_by_action_id[action]
        before_features = self._board_features
        after_board = clone_board(outcome.board_after)
        after_features = outcome.features

        lines_cleared = int(outcome.lines_cleared)
        score_delta = self._score_delta(lines_cleared)
        eroded_piece_cells = int(outcome.eroded_piece_cells)

        self.total_score += score_delta
        reward = self._compute_reward(
            before_features=before_features,
            after_features=after_features,
            lines_cleared=lines_cleared,
            eroded_piece_cells=eroded_piece_cells,
        )

        self.board = after_board
        self._board_features = after_features
        self.total_lines_cleared += lines_cleared
        self.current_piece = self.next_piece
        self.next_piece = self._pop_piece()
        self._refresh_candidates()

        if not np.any(self._mask):
            terminated = True
            reward += float(self.reward_config.game_over_penalty)
        if self.step_count >= self.max_steps:
            truncated = True

        obs = self._get_obs()
        info = self._get_info()
        info["selected_action"] = int(action)
        info["lines_cleared_this_step"] = lines_cleared
        info["score_delta"] = float(score_delta)
        info["eroded_piece_cells"] = eroded_piece_cells
        info["invalid_action"] = False
        return obs, float(reward), terminated, truncated, info

    def render(self) -> str | np.ndarray | None:
        if self.render_mode == "ansi":
            return (
                f"step={self.step_count} current={self.current_piece} next={self.next_piece} "
                f"total_lines={self.total_lines_cleared} total_score={self.total_score:.1f}\n"
                f"{board_to_ascii(self.board)}"
            )
        if self.render_mode == "rgb_array":
            return self._render_rgb_array()
        return None

    def close(self) -> None:
        pass

    def action_masks(self) -> np.ndarray:
        return self._mask.copy()

    def has_valid_actions(self) -> bool:
        return bool(np.any(self._mask))

    def valid_action_ids(self) -> list[int]:
        return [int(i) for i in np.flatnonzero(self._mask)]

    def _get_obs(self) -> dict[str, Any]:
        return {
            "board": self.board.astype(np.uint8, copy=True),
            "current_piece": int(PIECE_TO_INDEX[self.current_piece]),
            "next_piece": int(PIECE_TO_INDEX[self.next_piece]),
            "action_mask": self._mask.astype(np.uint8, copy=True),
        }

    def _get_info(self) -> dict[str, Any]:
        return {
            "current_piece": self.current_piece,
            "next_piece": self.next_piece,
            "valid_action_count": int(np.sum(self._mask)),
            "step_count": int(self.step_count),
            "total_lines_cleared": int(self.total_lines_cleared),
            "total_score": float(self.total_score),
        }

    def _refresh_candidates(self) -> None:
        mask, candidates, outcomes = enumerate_candidates(self.board, self.current_piece)
        self._mask = mask
        self._candidates = candidates
        self._outcomes = outcomes
        self._outcome_by_action_id = {outcome.candidate.action_id: outcome for outcome in outcomes}
        self._board_features = extract_features(self.board)

    def _score_delta(self, lines_cleared: int) -> float:
        rc = self.reward_config
        if lines_cleared <= 0:
            return 0.0
        if lines_cleared == 1:
            return float(rc.single_score)
        if lines_cleared == 2:
            return float(rc.double_score)
        if lines_cleared == 3:
            return float(rc.triple_score)
        return float(rc.tetris_score)

    def _potential_from_features(self, feats: PlacementFeatures) -> float:
        rc = self.reward_config
        return -(
            rc.aggregate_height_weight * float(feats.aggregate_height)
            + rc.holes_weight * float(feats.holes)
            + rc.bumpiness_weight * float(feats.bumpiness)
            + _max_height_penalty_from_features(feats, rc.max_height_weight)
            + rc.row_transitions_weight * float(feats.row_transitions)
            + rc.column_transitions_weight * float(feats.column_transitions)
            + rc.rows_with_holes_weight * float(feats.rows_with_holes)
            + rc.hole_depth_weight * float(feats.hole_depth)
            + rc.cumulative_wells_weight * float(feats.cumulative_wells)
        )

    def _potential(self, board: np.ndarray) -> float:
        return self._potential_from_features(extract_features(board))

    def _compute_reward(
        self,
        *,
        before_features: PlacementFeatures,
        after_features: PlacementFeatures,
        lines_cleared: int,
        eroded_piece_cells: int,
    ) -> float:
        rc = self.reward_config
        score_delta = self._score_delta(lines_cleared)
        phi_before = self._potential_from_features(before_features)
        phi_after = self._potential_from_features(after_features)
        reward = 0.0
        reward += score_delta
        reward += rc.eroded_piece_cells_weight * float(eroded_piece_cells)
        reward += rc.shaping_scale * (rc.shaping_gamma * phi_after - phi_before)
        reward += rc.survival_bonus
        return reward

    def _pop_piece(self) -> PieceId:
        if not self._bag:
            self._bag = list(PIECES)
            self.np_random.shuffle(self._bag)
        return self._bag.pop()

    def _render_rgb_array(self) -> np.ndarray:
        upscale = self.render_upscale
        empty_color = np.array([15, 16, 26], dtype=np.uint8)
        filled_color = np.array([0, 240, 255], dtype=np.uint8)
        grid_color = np.array([45, 45, 60], dtype=np.uint8)
        img = np.zeros((BOARD_HEIGHT, BOARD_WIDTH, 3), dtype=np.uint8)
        img[:, :] = empty_color
        img[self.board != 0] = filled_color
        frame = np.repeat(np.repeat(img, upscale, axis=0), upscale, axis=1)
        frame[::upscale, :, :] = grid_color
        frame[:, ::upscale, :] = grid_color
        return frame
