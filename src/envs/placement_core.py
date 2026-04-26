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
    row_transitions: int
    column_transitions: int
    rows_with_holes: int
    hole_depth: int
    cumulative_wells: int


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
    eroded_piece_cells: int
    features: PlacementFeatures


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
            if len(cells_local) != 4:
                raise ValueError(
                    f"Piece {piece_id} orientation {orientation_id}: expected 4 cells, got {len(cells_local)}"
                )
            if any(x < 0 or y < 0 for x, y in cells_local):
                raise ValueError(
                    f"Piece {piece_id} orientation {orientation_id}: negative coordinates after normalization"
                )

            width = 1 + max(x for x, _ in cells_local)
            height = 1 + max(y for _, y in cells_local)
            if width > BOARD_WIDTH:
                raise ValueError(
                    f"Piece {piece_id} orientation {orientation_id}: width {width} exceeds board width {BOARD_WIDTH}"
                )

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
PIECE_ORIENTATION_BY_ID = {
    piece_id: {ori.orientation_id: ori for ori in orientations}
    for piece_id, orientations in PIECE_LIBRARY.items()
}


# Precompute the bottom-most occupied dy for each local column in each orientation.
ORIENTATION_COLUMN_BOTTOMS: Dict[tuple[str, int], Tuple[Tuple[int, int], ...]] = {}
for _piece_id, _orientations in PIECE_LIBRARY.items():
    for _orientation in _orientations:
        bottoms: Dict[int, int] = {}
        for dx, dy in _orientation.cells_local:
            bottoms[dx] = max(bottoms.get(dx, -10_000), dy)
        ORIENTATION_COLUMN_BOTTOMS[(_piece_id, _orientation.orientation_id)] = tuple(sorted(bottoms.items()))


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
assert ACTION_SPACE_SIZE == 34, f"Expected ACTION_SPACE_SIZE=34, got {ACTION_SPACE_SIZE}"


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

    presets_dir = Path(__file__).resolve().parent.parent.parent / "data" / "boards"
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
                raise ValueError(f"Invalid character {ch!r} at row {y + 1}, col {x + 1} in {board_file}")
    return board


def occupancy_encoding(board: np.ndarray) -> np.ndarray:
    return (board != 0).astype(np.float32)


def world_cells(orientation: PieceOrientation, x: int, y: int) -> Tuple[Cell, ...]:
    return tuple((x + dx, y + dy) for dx, dy in orientation.cells_local)


def collides(board: np.ndarray, orientation: PieceOrientation, x: int, y: int) -> bool:
    for dx, dy in orientation.cells_local:
        wx = x + dx
        wy = y + dy
        if wx < 0 or wx >= BOARD_WIDTH or wy >= BOARD_HEIGHT:
            return True
        if wy >= 0 and board[wy, wx] != 0:
            return True
    return False


def _compute_board_surface(board: np.ndarray) -> np.ndarray:
    surface = np.full(BOARD_WIDTH, BOARD_HEIGHT, dtype=np.int32)
    for x in range(BOARD_WIDTH):
        for y in range(BOARD_HEIGHT):
            if board[y, x] != 0:
                surface[x] = y
                break
    return surface


def compute_drop_y_fast(board: np.ndarray, surface: np.ndarray, orientation: PieceOrientation, x: int) -> int | None:
    # Compute the maximum legal y directly from column surfaces.
    y_final = BOARD_HEIGHT
    for dx, bottom_dy in ORIENTATION_COLUMN_BOTTOMS[(orientation.piece_id, orientation.orientation_id)]:
        col = x + dx
        if col < 0 or col >= BOARD_WIDTH:
            return None
        candidate_y = int(surface[col]) - bottom_dy - 1
        if candidate_y < y_final:
            y_final = candidate_y

    if y_final == BOARD_HEIGHT:
        return None
    if collides(board, orientation, x, y_final):
        return None
    for _, dy in orientation.cells_local:
        if y_final + dy < 0:
            return None
    return int(y_final)


def compute_drop_y(board: np.ndarray, orientation: PieceOrientation, x: int) -> int | None:
    # Keep the public function name, but route to the fast implementation.
    surface = _compute_board_surface(board)
    return compute_drop_y_fast(board, surface, orientation, x)


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
    return collides(board, orientation, default_spawn_x(piece_id), default_spawn_y(piece_id))


def lock_piece(board: np.ndarray, cells_world: Iterable[Cell], fill_value: int = 1) -> np.ndarray:
    board_after = clone_board(board)
    for x, y in cells_world:
        board_after[y, x] = fill_value
    return board_after


def _lock_piece_inplace(board: np.ndarray, cells_world: Iterable[Cell], fill_value: int = 1) -> None:
    for x, y in cells_world:
        board[y, x] = fill_value


def clear_full_lines(board: np.ndarray) -> Tuple[np.ndarray, int]:
    full_rows = np.all(board != 0, axis=1)
    lines_cleared = int(np.sum(full_rows))
    if lines_cleared == 0:
        return clone_board(board), 0
    remaining = board[~full_rows]
    cleared_board = np.zeros_like(board)
    cleared_board[lines_cleared:, :] = remaining
    return cleared_board, lines_cleared


def apply_placement_owned(board: np.ndarray, cells_world: Iterable[Cell], fill_value: int = 1) -> Tuple[np.ndarray, int, int]:
    # board is already owned by the caller and may be mutated in-place.
    cell_rows = [0, 0, 0, 0]
    i = 0
    for x, y in cells_world:
        board[y, x] = fill_value
        cell_rows[i] = y
        i += 1

    full_rows = np.all(board != 0, axis=1)
    lines_cleared = int(np.sum(full_rows))

    if lines_cleared > 0:
        eroded_piece_cells = lines_cleared * sum(1 for y in cell_rows[:i] if full_rows[y])
        remaining = board[~full_rows]
        board[:, :] = 0
        board[lines_cleared:, :] = remaining
    else:
        eroded_piece_cells = 0

    return board, lines_cleared, int(eroded_piece_cells)


def apply_placement(board: np.ndarray, cells_world: Iterable[Cell], fill_value: int = 1) -> Tuple[np.ndarray, int]:
    board_after = clone_board(board)
    board_after, lines_cleared, _ = apply_placement_owned(board_after, cells_world, fill_value=fill_value)
    return board_after, lines_cleared


def column_heights(board: np.ndarray) -> np.ndarray:
    heights = np.zeros(BOARD_WIDTH, dtype=np.int32)
    for x in range(BOARD_WIDTH):
        for y in range(BOARD_HEIGHT):
            if board[y, x] != 0:
                heights[x] = BOARD_HEIGHT - y
                break
    return heights


def hole_mask(board: np.ndarray) -> np.ndarray:
    mask = np.zeros_like(board, dtype=np.uint8)
    for x in range(BOARD_WIDTH):
        filled_seen = False
        for y in range(BOARD_HEIGHT):
            cell_val = board[y, x]
            if cell_val != 0:
                filled_seen = True
            elif filled_seen:
                mask[y, x] = 1
    return mask


def count_holes(board: np.ndarray) -> int:
    return int(np.sum(hole_mask(board)))


def row_transitions(board: np.ndarray) -> int:
    transitions = 0
    for y in range(BOARD_HEIGHT):
        prev_filled = 1
        for x in range(BOARD_WIDTH):
            cur_filled = 1 if board[y, x] != 0 else 0
            if cur_filled != prev_filled:
                transitions += 1
            prev_filled = cur_filled
        if prev_filled == 0:
            transitions += 1
    return transitions


def column_transitions(board: np.ndarray) -> int:
    transitions = 0
    for x in range(BOARD_WIDTH):
        prev_filled = 1
        for y in range(BOARD_HEIGHT):
            cur_filled = 1 if board[y, x] != 0 else 0
            if cur_filled != prev_filled:
                transitions += 1
            prev_filled = cur_filled
        if prev_filled == 0:
            transitions += 1
    return transitions


def rows_with_holes(board: np.ndarray) -> int:
    return int(np.count_nonzero(np.any(hole_mask(board) != 0, axis=1)))


def hole_depth(board: np.ndarray) -> int:
    total = 0
    for x in range(BOARD_WIDTH):
        filled_above = 0
        seen_block = False
        for y in range(BOARD_HEIGHT):
            if board[y, x] != 0:
                filled_above += 1
                seen_block = True
            elif seen_block:
                total += filled_above
    return total


def cumulative_wells(board: np.ndarray) -> int:
    total = 0
    for x in range(BOARD_WIDTH):
        depth = 0
        for y in range(BOARD_HEIGHT):
            if board[y, x] != 0:
                depth = 0
                continue
            left_filled = (x == 0) or (board[y, x - 1] != 0)
            right_filled = (x == BOARD_WIDTH - 1) or (board[y, x + 1] != 0)
            if left_filled and right_filled:
                depth += 1
                total += depth
            else:
                depth = 0
    return total


def count_eroded_piece_cells(before_board: np.ndarray, candidate_cells: Tuple[Cell, ...], lines_cleared: int) -> int:
    if lines_cleared <= 0:
        return 0
    owned = clone_board(before_board)
    _, _, eroded = apply_placement_owned(owned, candidate_cells, fill_value=1)
    return int(eroded)


def extract_features(board: np.ndarray) -> PlacementFeatures:
    heights = np.zeros(BOARD_WIDTH, dtype=np.int32)
    row_has_holes = np.zeros(BOARD_HEIGHT, dtype=bool)
    holes = 0
    aggregate_height = 0
    max_height = 0
    hole_depth_total = 0
    column_transitions_total = 0

    # Pass 1: column-wise features (heights, holes, depth, column transitions)
    for x in range(BOARD_WIDTH):
        filled_seen = False
        filled_above = 0
        col_height = 0
        prev_filled = 1
        for y in range(BOARD_HEIGHT):
            cur_filled = 1 if board[y, x] != 0 else 0
            if cur_filled != prev_filled:
                column_transitions_total += 1
            prev_filled = cur_filled

            if cur_filled:
                filled_above += 1
                if not filled_seen:
                    col_height = BOARD_HEIGHT - y
                    filled_seen = True
                    if col_height > max_height:
                        max_height = col_height
            elif filled_seen:
                holes += 1
                row_has_holes[y] = True
                hole_depth_total += filled_above
        if prev_filled == 0:
            column_transitions_total += 1

        heights[x] = col_height
        aggregate_height += int(col_height)

    bumpiness_total = int(np.sum(np.abs(np.diff(heights)))) if BOARD_WIDTH > 1 else 0
    rows_with_holes_total = int(np.sum(row_has_holes))

    # Pass 2: row transitions
    row_transitions_total = 0
    for y in range(BOARD_HEIGHT):
        prev_filled = 1
        for x in range(BOARD_WIDTH):
            cur_filled = 1 if board[y, x] != 0 else 0
            if cur_filled != prev_filled:
                row_transitions_total += 1
            prev_filled = cur_filled
        if prev_filled == 0:
            row_transitions_total += 1

    # Pass 3: cumulative wells
    cumulative_wells_total = 0
    for x in range(BOARD_WIDTH):
        depth = 0
        for y in range(BOARD_HEIGHT):
            if board[y, x] != 0:
                depth = 0
                continue
            left_filled = (x == 0) or (board[y, x - 1] != 0)
            right_filled = (x == BOARD_WIDTH - 1) or (board[y, x + 1] != 0)
            if left_filled and right_filled:
                depth += 1
                cumulative_wells_total += depth
            else:
                depth = 0

    return PlacementFeatures(
        aggregate_height=int(aggregate_height),
        holes=int(holes),
        bumpiness=int(bumpiness_total),
        max_height=int(max_height),
        row_transitions=int(row_transitions_total),
        column_transitions=int(column_transitions_total),
        rows_with_holes=int(rows_with_holes_total),
        hole_depth=int(hole_depth_total),
        cumulative_wells=int(cumulative_wells_total),
    )


def enumerate_candidates(board: np.ndarray, piece_id: PieceId) -> Tuple[np.ndarray, List[CandidatePlacement], List[PlacementOutcome]]:
    if piece_id not in PIECE_LIBRARY:
        raise ValueError(f"Unknown piece_id: {piece_id}")

    mask = np.zeros(ACTION_SPACE_SIZE, dtype=bool)
    candidates: List[CandidatePlacement] = []
    outcomes: List[PlacementOutcome] = []

    surface = _compute_board_surface(board)
    slots = CANONICAL_SLOTS[piece_id]
    orientation_lookup = PIECE_ORIENTATION_BY_ID[piece_id]

    for slot in slots:
        orientation = orientation_lookup[slot.orientation_id]
        y_final = compute_drop_y_fast(board, surface, orientation, slot.x)
        if y_final is None:
            continue

        cells = tuple((slot.x + dx, y_final + dy) for dx, dy in orientation.cells_local)
        candidate = CandidatePlacement(
            action_id=slot.action_id,
            piece_id=piece_id,
            orientation_id=slot.orientation_id,
            x=slot.x,
            y_final=y_final,
            cells_world=cells,
        )

        board_after = clone_board(board)
        board_after, lines_cleared, eroded = apply_placement_owned(board_after, cells, fill_value=1)
        features = extract_features(board_after)

        outcome = PlacementOutcome(
            candidate=candidate,
            board_after=board_after,
            lines_cleared=int(lines_cleared),
            eroded_piece_cells=int(eroded),
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
