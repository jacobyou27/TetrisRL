from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np

BOARD_HEIGHT = 20
BOARD_WIDTH = 10

PieceId = str
Cell = Tuple[int, int]


@dataclass(frozen=True)
class PieceOrientation:
    piece_id: PieceId
    orientation_id: int
    cells_local: Tuple[Cell, ...]
    width: int
    height: int


@dataclass(frozen=True)
class PlacementSlot:
    action_id: int
    piece_id: PieceId
    orientation_id: int
    x: int


@dataclass
class PlacementFeatures:
    aggregate_height: int
    holes: int
    bumpiness: int
    max_height: int


@dataclass
class CandidatePlacement:
    action_id: int
    piece_id: PieceId
    orientation_id: int
    x: int
    y_final: int
    cells_world: Tuple[Cell, ...]


@dataclass
class PlacementOutcome:
    candidate: CandidatePlacement
    board_after: np.ndarray
    lines_cleared: int
    features: PlacementFeatures


# Unique orientation footprints, normalized to top-left local origin.
# This is a placement engine, not a controller-accurate simulator.
_RAW_PIECES: Dict[PieceId, Tuple[Tuple[Cell, ...], ...]] = {
    "I": (
        ((0, 0), (1, 0), (2, 0), (3, 0)),
        ((0, 0), (0, 1), (0, 2), (0, 3)),
    ),
    "O": (
        ((0, 0), (1, 0), (0, 1), (1, 1)),
    ),
    "T": (
        ((0, 0), (1, 0), (2, 0), (1, 1)),
        ((1, 0), (0, 1), (1, 1), (1, 2)),
        ((1, 0), (0, 1), (1, 1), (2, 1)),
        ((0, 0), (0, 1), (1, 1), (0, 2)),
    ),
    "J": (
        ((0, 0), (0, 1), (1, 1), (2, 1)),
        ((0, 0), (1, 0), (0, 1), (0, 2)),
        ((0, 0), (1, 0), (2, 0), (2, 1)),
        ((1, 0), (1, 1), (0, 2), (1, 2)),
    ),
    "L": (
        ((2, 0), (0, 1), (1, 1), (2, 1)),
        ((0, 0), (0, 1), (0, 2), (1, 2)),
        ((0, 0), (1, 0), (2, 0), (0, 1)),
        ((0, 0), (1, 0), (1, 1), (1, 2)),
    ),
    "S": (
        ((1, 0), (2, 0), (0, 1), (1, 1)),
        ((0, 0), (0, 1), (1, 1), (1, 2)),
    ),
    "Z": (
        ((0, 0), (1, 0), (1, 1), (2, 1)),
        ((1, 0), (0, 1), (1, 1), (0, 2)),
    ),
}


def _normalize_cells(cells: Iterable[Cell]) -> Tuple[Cell, ...]:
    xs = [x for x, _ in cells]
    ys = [y for _, y in cells]
    min_x = min(xs)
    min_y = min(ys)
    return tuple(sorted((x - min_x, y - min_y) for x, y in cells))


def _build_piece_library() -> Dict[PieceId, Tuple[PieceOrientation, ...]]:
    library: Dict[PieceId, Tuple[PieceOrientation, ...]] = {}
    for piece_id, raw_orientations in _RAW_PIECES.items():
        orientations: List[PieceOrientation] = []
        for orientation_id, raw_cells in enumerate(raw_orientations):
            cells_local = _normalize_cells(raw_cells)
            width = 1 + max(x for x, _ in cells_local)
            height = 1 + max(y for _, y in cells_local)
            orientations.append(
                PieceOrientation(
                    piece_id=piece_id,
                    orientation_id=orientation_id,
                    cells_local=cells_local,
                    width=width,
                    height=height,
                )
            )
        library[piece_id] = tuple(orientations)
    return library


PIECE_LIBRARY = _build_piece_library()


def _build_canonical_slots(
    piece_library: Dict[PieceId, Tuple[PieceOrientation, ...]],
) -> Dict[PieceId, Tuple[PlacementSlot, ...]]:
    slots_per_piece: Dict[PieceId, Tuple[PlacementSlot, ...]] = {}

    for piece_id, orientations in piece_library.items():
        slots: List[PlacementSlot] = []
        next_action_id = 0

        for orientation in orientations:
            max_x = BOARD_WIDTH - orientation.width
            for x in range(max_x + 1):
                slots.append(
                    PlacementSlot(
                        action_id=next_action_id,
                        piece_id=piece_id,
                        orientation_id=orientation.orientation_id,
                        x=x,
                    )
                )
                next_action_id += 1

        slots_per_piece[piece_id] = tuple(slots)

    return slots_per_piece


CANONICAL_SLOTS = _build_canonical_slots(PIECE_LIBRARY)
ACTION_SPACE_SIZE = max(len(slots) for slots in CANONICAL_SLOTS.values())


def empty_board() -> np.ndarray:
    return np.zeros((BOARD_HEIGHT, BOARD_WIDTH), dtype=np.uint8)


def clone_board(board: np.ndarray) -> np.ndarray:
    return np.array(board, copy=True)


def board_to_ascii(board: np.ndarray) -> str:
    rows = []
    for y in range(BOARD_HEIGHT):
        row = "".join("#" if board[y, x] else "." for x in range(BOARD_WIDTH))
        rows.append(row)
    return "\n".join(rows)


def make_preset_board(name: str) -> np.ndarray:
    from pathlib import Path
    
    presets_dir = Path(__file__).resolve().parent.parent.parent / "boards"
    board_file = presets_dir / f"{name}.txt"
    
    if not board_file.exists():
        raise ValueError(f"Unknown preset board: {name}")
    
    board_text = board_file.read_text(encoding="utf-8")
    board = empty_board()
    
    lines = [line.rstrip("\n") for line in board_text.splitlines()]
    lines = [line for line in lines if line.strip() != ""]
    
    if len(lines) != BOARD_HEIGHT:
        raise ValueError(f"Board file must contain exactly {BOARD_HEIGHT} lines: {board_file}")
    
    for y, line in enumerate(lines):
        if len(line) != BOARD_WIDTH:
            raise ValueError(f"Line {y + 1} in {board_file} must contain exactly {BOARD_WIDTH} characters")
        
        for x, ch in enumerate(line):
            if ch in {"#", "X", "1", "@"}:
                board[y, x] = 1
            elif ch not in {".", "0", "_", " "}:
                raise ValueError(
                    f"Invalid character '{ch}' at row {y + 1}, col {x + 1} in {board_file}"
                )
    
    return board


def occupancy_encoding(board: np.ndarray) -> np.ndarray:
    return board.astype(np.float32)


def world_cells(orientation: PieceOrientation, x: int, y: int) -> Tuple[Cell, ...]:
    return tuple((x + dx, y + dy) for dx, dy in orientation.cells_local)


def collides(board: np.ndarray, orientation: PieceOrientation, x: int, y: int) -> bool:
    for wx, wy in world_cells(orientation, x, y):
        if wx < 0 or wx >= BOARD_WIDTH:
            return True
        if wy >= BOARD_HEIGHT:
            return True
        if wy >= 0 and board[wy, wx] != 0:
            return True
    return False


def compute_drop_y(board: np.ndarray, orientation: PieceOrientation, x: int) -> int | None:
    y = -orientation.height

    while not collides(board, orientation, x, y + 1):
        y += 1

    if collides(board, orientation, x, y):
        return None

    final_cells = world_cells(orientation, x, y)
    if any(wy < 0 for _, wy in final_cells):
        return None

    return y


def default_spawn_orientation(piece_id: PieceId) -> PieceOrientation:
    return PIECE_LIBRARY[piece_id][0]


def default_spawn_x(piece_id: PieceId) -> int:
    orientation = default_spawn_orientation(piece_id)
    return (BOARD_WIDTH - orientation.width) // 2


def default_spawn_y(piece_id: PieceId) -> int:
    orientation = default_spawn_orientation(piece_id)
    return -orientation.height


def spawn_overlaps_visible_stack(board: np.ndarray, piece_id: PieceId) -> bool:
    orientation = default_spawn_orientation(piece_id)
    x = default_spawn_x(piece_id)
    y = default_spawn_y(piece_id)
    return collides(board, orientation, x, y)


def lock_piece(board: np.ndarray, cells_world: Iterable[Cell], fill_value: int = 1) -> np.ndarray:
    board_after = clone_board(board)
    for x, y in cells_world:
        board_after[y, x] = fill_value
    return board_after


def clear_full_lines(board: np.ndarray) -> Tuple[np.ndarray, int]:
    full_rows = np.all(board != 0, axis=1)
    lines_cleared = int(np.sum(full_rows))

    if lines_cleared == 0:
        return clone_board(board), 0

    remaining = board[~full_rows]
    padding = np.zeros((lines_cleared, BOARD_WIDTH), dtype=board.dtype)
    cleared_board = np.vstack([padding, remaining])
    return cleared_board, lines_cleared


def apply_placement(board: np.ndarray, cells_world: Iterable[Cell], fill_value: int = 1) -> Tuple[np.ndarray, int]:
    locked = lock_piece(board, cells_world, fill_value=fill_value)
    return clear_full_lines(locked)


def column_heights(board: np.ndarray) -> np.ndarray:
    heights = np.zeros(BOARD_WIDTH, dtype=np.int32)

    for x in range(BOARD_WIDTH):
        filled_rows = np.where(board[:, x] != 0)[0]
        if filled_rows.size > 0:
            heights[x] = BOARD_HEIGHT - int(filled_rows[0])

    return heights


def hole_mask(board: np.ndarray) -> np.ndarray:
    mask = np.zeros_like(board, dtype=np.uint8)

    for x in range(BOARD_WIDTH):
        filled_seen = False
        for y in range(BOARD_HEIGHT):
            if board[y, x] != 0:
                filled_seen = True
            elif filled_seen:
                mask[y, x] = 1

    return mask


def count_holes(board: np.ndarray) -> int:
    return int(np.sum(hole_mask(board)))


def extract_features(board: np.ndarray) -> PlacementFeatures:
    heights = column_heights(board)
    return PlacementFeatures(
        aggregate_height=int(np.sum(heights)),
        holes=count_holes(board),
        bumpiness=int(np.sum(np.abs(np.diff(heights)))),
        max_height=int(np.max(heights)) if heights.size > 0 else 0,
    )


def enumerate_candidates(
    board: np.ndarray,
    piece_id: PieceId,
) -> Tuple[np.ndarray, List[CandidatePlacement], List[PlacementOutcome]]:
    if piece_id not in PIECE_LIBRARY:
        raise ValueError(f"Unknown piece_id: {piece_id}")

    mask = np.zeros(ACTION_SPACE_SIZE, dtype=bool)
    candidates: List[CandidatePlacement] = []
    outcomes: List[PlacementOutcome] = []

    orientations = PIECE_LIBRARY[piece_id]
    slots = CANONICAL_SLOTS[piece_id]
    orientation_lookup = {ori.orientation_id: ori for ori in orientations}

    for slot in slots:
        orientation = orientation_lookup[slot.orientation_id]
        y_final = compute_drop_y(board, orientation, slot.x)

        if y_final is None:
            continue

        cells = world_cells(orientation, slot.x, y_final)
        candidate = CandidatePlacement(
            action_id=slot.action_id,
            piece_id=piece_id,
            orientation_id=slot.orientation_id,
            x=slot.x,
            y_final=y_final,
            cells_world=cells,
        )

        board_after, lines_cleared = apply_placement(board, cells, fill_value=1)
        features = extract_features(board_after)

        outcome = PlacementOutcome(
            candidate=candidate,
            board_after=board_after,
            lines_cleared=lines_cleared,
            features=features,
        )

        mask[slot.action_id] = True
        candidates.append(candidate)
        outcomes.append(outcome)

    return mask, candidates, outcomes


def candidate_by_action_id(candidates: Iterable[CandidatePlacement], action_id: int) -> CandidatePlacement | None:
    for candidate in candidates:
        if candidate.action_id == action_id:
            return candidate
    return None


def outcome_by_action_id(outcomes: Iterable[PlacementOutcome], action_id: int) -> PlacementOutcome | None:
    for outcome in outcomes:
        if outcome.candidate.action_id == action_id:
            return outcome
    return None


def apply_action(board: np.ndarray, piece_id: PieceId, action_id: int) -> PlacementOutcome:
    mask, _, outcomes = enumerate_candidates(board, piece_id)

    if action_id < 0 or action_id >= ACTION_SPACE_SIZE:
        raise ValueError(f"action_id must be in [0, {ACTION_SPACE_SIZE - 1}]")

    if not mask[action_id]:
        raise ValueError(f"Invalid action_id {action_id} for piece {piece_id} on this board")

    outcome = outcome_by_action_id(outcomes, action_id)
    if outcome is None:
        raise RuntimeError("Action mask and outcomes are inconsistent")

    return outcome


def has_any_valid_action(board: np.ndarray, piece_id: PieceId) -> bool:
    mask, _, _ = enumerate_candidates(board, piece_id)
    return bool(np.any(mask))