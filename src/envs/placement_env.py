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
    line_clear_weight: float = 1.0
    aggregate_height_weight: float = -0.03
    holes_weight: float = -0.15
    bumpiness_weight: float = -0.03
    max_height_weight: float = 0.0
    survival_bonus: float = 0.0
    game_over_penalty: float = -5.0
    invalid_action_penalty: float = -2.0


class PlacementTetrisEnv(gym.Env):
    """
    Placement-based Tetris environment.

    Action:
        Discrete slot id in [0, ACTION_SPACE_SIZE - 1]

    Observation:
        Dict with:
            - board: (20, 10) uint8 occupancy grid
            - current_piece: scalar int in [0, 6]
            - next_piece: scalar int in [0, 6]
            - action_mask: (ACTION_SPACE_SIZE,) uint8 mask

    Notes:
        - This is NOT controller-accurate Tetris.
        - Actions choose final drop-only placements.
        - Overhang tucks / cavity entries are intentionally ignored.
    """

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
                "board": spaces.Box(
                    low=0,
                    high=1,
                    shape=(BOARD_HEIGHT, BOARD_WIDTH),
                    dtype=np.uint8,
                ),
                "current_piece": spaces.Discrete(len(PIECES)),
                "next_piece": spaces.Discrete(len(PIECES)),
                "action_mask": spaces.Box(
                    low=0,
                    high=1,
                    shape=(ACTION_SPACE_SIZE,),
                    dtype=np.uint8,
                ),
            }
        )

        self.board: np.ndarray = empty_board()
        self.current_piece: PieceId = "I"
        self.next_piece: PieceId = "O"
        self.step_count: int = 0
        self.total_lines_cleared: int = 0
        self._bag: list[PieceId] = []

        self._mask = np.zeros(ACTION_SPACE_SIZE, dtype=bool)
        self._candidates = []
        self._outcomes = []
        self._outcome_by_action_id: dict[int, PlacementOutcome] = {}

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        super().reset(seed=seed)

        options = options or {}
        self.step_count = 0
        self.total_lines_cleared = 0

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
        return obs, info

    def step(
        self,
        action: int,
    ) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
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
            return self._get_obs(), reward, terminated, truncated, info

        before_features = extract_features(self.board)
        outcome = self._outcome_by_action_id[action]

        self.board = clone_board(outcome.board_after)
        self.total_lines_cleared += int(outcome.lines_cleared)

        reward = self._compute_reward(
            before_features=before_features,
            outcome=outcome,
        )

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
        info["lines_cleared_this_step"] = int(outcome.lines_cleared)
        info["invalid_action"] = False

        return obs, float(reward), terminated, truncated, info

    def render(self) -> str | np.ndarray | None:
        if self.render_mode == "ansi":
            return (
                f"step={self.step_count} "
                f"current={self.current_piece} "
                f"next={self.next_piece} "
                f"total_lines={self.total_lines_cleared}\n"
                f"{board_to_ascii(self.board)}"
            )

        if self.render_mode == "rgb_array":
            return self._render_rgb_array()

        return None

    def close(self) -> None:
        pass

    def action_masks(self) -> np.ndarray:
        """
        Returns a boolean mask over the discrete action space.
        True means the placement slot is valid for the current piece on the current board.
        """
        return self._mask.copy()

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
        }

    def _refresh_candidates(self) -> None:
        mask, candidates, outcomes = enumerate_candidates(self.board, self.current_piece)
        self._mask = mask
        self._candidates = candidates
        self._outcomes = outcomes
        self._outcome_by_action_id = {
            outcome.candidate.action_id: outcome for outcome in outcomes
        }

    def _compute_reward(
        self,
        *,
        before_features,
        outcome: PlacementOutcome,
    ) -> float:
        after = outcome.features
        rc = self.reward_config

        reward = 0.0
        reward += rc.line_clear_weight * float(outcome.lines_cleared ** 2)
        reward += rc.aggregate_height_weight * float(after.aggregate_height)
        reward += rc.holes_weight * float(after.holes)
        reward += rc.bumpiness_weight * float(after.bumpiness)
        reward += rc.max_height_weight * float(after.max_height)
        reward += rc.survival_bonus

        return reward

    def _pop_piece(self) -> PieceId:
        if not self._bag:
            self._bag = list(PIECES)
            self.np_random.shuffle(self._bag)
        return self._bag.pop()

    def _render_rgb_array(self) -> np.ndarray:
        """
        Simple RGB renderer for the logical 20x10 placement board.
        This is a state renderer, not a falling-piece animation.
        """
        upscale = self.render_upscale

        # Base colors
        empty_color = np.array([18, 18, 18], dtype=np.uint8)
        filled_color = np.array([0, 200, 255], dtype=np.uint8)
        grid_color = np.array([45, 45, 45], dtype=np.uint8)

        img = np.zeros((BOARD_HEIGHT, BOARD_WIDTH, 3), dtype=np.uint8)
        img[:, :] = empty_color
        img[self.board != 0] = filled_color

        # Upscale cells
        frame = np.repeat(np.repeat(img, upscale, axis=0), upscale, axis=1)

        # Draw grid lines
        frame[::upscale, :, :] = grid_color
        frame[:, ::upscale, :] = grid_color

        return frame