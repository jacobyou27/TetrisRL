from __future__ import annotations

import argparse
import sys
from pathlib import Path
import tkinter as tk
from tkinter import ttk, simpledialog, messagebox
from tkinter.scrolledtext import ScrolledText

import matplotlib
matplotlib.use("TkAgg")

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import matplotlib.patches as patches

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.envs.placement_core import (
    BOARD_HEIGHT,
    BOARD_WIDTH,
    ACTION_SPACE_SIZE,
    candidate_by_action_id,
    default_spawn_orientation,
    default_spawn_x,
    default_spawn_y,
    enumerate_candidates,
    has_any_valid_action,
    hole_mask,
    occupancy_encoding,
    outcome_by_action_id,
    spawn_overlaps_visible_stack,
)

PIECES = ["I", "O", "T", "J", "L", "S", "Z"]
ENCODINGS = ["occupancy", "heights", "holes", "candidates", "result"]
BOARDS_DIR = PROJECT_ROOT / "boards"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interactive placement tester GUI.")
    parser.add_argument("--preset", type=str, default="overhang")
    parser.add_argument("--piece", type=str, default="T", choices=PIECES)
    parser.add_argument("--encoding", type=str, default="occupancy", choices=ENCODINGS)
    return parser.parse_args()


def ensure_boards_dir() -> None:
    BOARDS_DIR.mkdir(parents=True, exist_ok=True)


def list_board_templates() -> list[str]:
    ensure_boards_dir()
    presets = [p.stem for p in sorted(BOARDS_DIR.glob("*.txt"))]
    return presets


def board_to_text(board: np.ndarray) -> str:
    lines = []
    for y in range(BOARD_HEIGHT):
        line = "".join("#" if board[y, x] else "." for x in range(BOARD_WIDTH))
        lines.append(line)
    return "\n".join(lines)


def parse_board_text(text: str) -> np.ndarray:
    raw_lines = [line.rstrip("\n") for line in text.splitlines()]
    lines = [line for line in raw_lines if line.strip() != ""]

    if len(lines) != BOARD_HEIGHT:
        raise ValueError(f"Board text must contain exactly {BOARD_HEIGHT} non-empty lines.")

    board = np.zeros((BOARD_HEIGHT, BOARD_WIDTH), dtype=np.uint8)

    for y, line in enumerate(lines):
        if len(line) != BOARD_WIDTH:
            raise ValueError(f"Line {y + 1} must contain exactly {BOARD_WIDTH} characters.")

        for x, ch in enumerate(line):
            if ch in {"#", "X", "1", "@"}:
                board[y, x] = 1
            elif ch in {".", "0", "_", " "}:
                board[y, x] = 0
            else:
                raise ValueError(
                    f"Invalid character '{ch}' at row {y + 1}, col {x + 1}. "
                    "Use '#' for filled and '.' for empty."
                )

    return board


def load_board_template(name: str) -> np.ndarray:
    ensure_boards_dir()
    path = BOARDS_DIR / f"{name}.txt"
    if not path.exists():
        raise FileNotFoundError(f"Board file not found: {path}")
    return parse_board_text(path.read_text(encoding="utf-8"))


def save_board_template(stem: str, board: np.ndarray) -> Path:
    ensure_boards_dir()
    safe_stem = "".join(ch for ch in stem if ch.isalnum() or ch in {"_", "-", "."}).strip(".")
    if not safe_stem:
        raise ValueError("Board name must contain at least one letter or number.")
    path = BOARDS_DIR / f"{safe_stem}.txt"
    path.write_text(board_to_text(board), encoding="utf-8")
    return path


def heights_encoding(board: np.ndarray) -> np.ndarray:
    heat = np.zeros_like(board, dtype=np.float32)
    for x in range(BOARD_WIDTH):
        filled_rows = np.where(board[:, x] != 0)[0]
        if filled_rows.size > 0:
            height = BOARD_HEIGHT - int(filled_rows[0])
            heat[BOARD_HEIGHT - height :, x] = 1.0
    return heat


def candidates_heatmap(candidates) -> np.ndarray:
    heat = np.zeros((BOARD_HEIGHT, BOARD_WIDTH), dtype=np.float32)
    for candidate in candidates:
        for x, y in candidate.cells_world:
            heat[y, x] += 1.0
    return heat


def selected_candidate_mask(candidate) -> np.ndarray:
    mask = np.zeros((BOARD_HEIGHT, BOARD_WIDTH), dtype=np.float32)
    if candidate is None:
        return mask
    for x, y in candidate.cells_world:
        mask[y, x] = 1.0
    return mask


def build_replay_scripts(piece_id: str, candidate) -> dict[str, str]:
    """
    These are illustrative replay scripts only.
    They are not controller-exact proofs of reachability.
    """
    spawn_orientation = default_spawn_orientation(piece_id)
    spawn_x = default_spawn_x(piece_id)
    spawn_y = default_spawn_y(piece_id)

    rot_steps = max(0, candidate.orientation_id - spawn_orientation.orientation_id)

    dx = candidate.x - spawn_x
    horizontal = []
    if dx < 0:
        horizontal = ["left"] * abs(dx)
    elif dx > 0:
        horizontal = ["right"] * dx

    rotations = ["rotate_cw"] * rot_steps
    drop_steps = max(0, candidate.y_final - spawn_y)
    drops = ["down"] * drop_steps

    interleaved = []
    h = horizontal.copy()
    d = drops.copy()
    while h or d:
        if h:
            interleaved.append(h.pop(0))
        if d:
            interleaved.append(d.pop(0))

    return {
        "rotate-first": " -> ".join(rotations + horizontal + drops) if (rotations or horizontal or drops) else "(no-op)",
        "move-first": " -> ".join(horizontal + rotations + drops) if (rotations or horizontal or drops) else "(no-op)",
        "interleaved": " -> ".join(rotations + interleaved) if (rotations or interleaved) else "(no-op)",
    }


class PlacementTesterApp:
    def __init__(self, root: tk.Tk, initial_preset: str, initial_piece: str, initial_encoding: str) -> None:
        self.root = root
        self.root.title("Tetris Placement Tester")
        self.root.geometry("1550x920")

        self.template_var = tk.StringVar(value=initial_preset)
        self.piece_var = tk.StringVar(value=initial_piece)
        self.encoding_var = tk.StringVar(value=initial_encoding)

        self.board = None
        self.mask = None
        self.candidates = []
        self.outcomes = []
        self.selected_action_id: int | None = None

        self._build_ui()
        self.refresh_templates()
        if initial_preset not in self.template_choices:
            self.template_var.set(self.template_choices[0])
        self.load_selected_template()

    def _build_ui(self) -> None:
        self.root.columnconfigure(0, weight=4)
        self.root.columnconfigure(1, weight=2)
        self.root.rowconfigure(1, weight=1)

        controls = ttk.Frame(self.root, padding=8)
        controls.grid(row=0, column=0, columnspan=2, sticky="ew")
        controls.columnconfigure(30, weight=1)

        ttk.Label(controls, text="Board").grid(row=0, column=0, padx=4)
        self.template_box = ttk.Combobox(controls, textvariable=self.template_var, state="readonly", width=18)
        self.template_box.grid(row=0, column=1, padx=4)
        self.template_box.bind("<<ComboboxSelected>>", lambda e: self.load_selected_template())

        ttk.Button(controls, text="Reload", command=self.load_selected_template).grid(row=0, column=2, padx=4)
        ttk.Button(controls, text="Clear Board", command=self.clear_board).grid(row=0, column=3, padx=4)
        ttk.Button(controls, text="Save Board As...", command=self.save_current_board).grid(row=0, column=4, padx=4)

        ttk.Label(controls, text="Piece").grid(row=0, column=5, padx=4)
        piece_box = ttk.Combobox(controls, textvariable=self.piece_var, values=PIECES, state="readonly", width=6)
        piece_box.grid(row=0, column=6, padx=4)
        piece_box.bind("<<ComboboxSelected>>", lambda e: self.recompute_only())

        ttk.Label(controls, text="Encoding").grid(row=0, column=7, padx=4)
        encoding_box = ttk.Combobox(controls, textvariable=self.encoding_var, values=ENCODINGS, state="readonly", width=12)
        encoding_box.grid(row=0, column=8, padx=4)
        encoding_box.bind("<<ComboboxSelected>>", lambda e: self.render())

        self.status_label = ttk.Label(controls, text="", anchor="w")
        self.status_label.grid(row=0, column=30, sticky="ew", padx=8)

        left = ttk.Frame(self.root, padding=(8, 0, 4, 8))
        left.grid(row=1, column=0, sticky="nsew")
        left.rowconfigure(0, weight=1)
        left.columnconfigure(0, weight=1)

        self.fig = Figure(figsize=(11, 7), dpi=100)
        self.ax_board = self.fig.add_subplot(1, 3, 1)
        self.ax_encoding = self.fig.add_subplot(1, 3, 2)
        self.ax_result = self.fig.add_subplot(1, 3, 3)

        self.canvas = FigureCanvasTkAgg(self.fig, master=left)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")
        self.canvas.mpl_connect("button_press_event", self.on_canvas_click)

        right = ttk.Frame(self.root, padding=(4, 0, 8, 8))
        right.grid(row=1, column=1, sticky="nsew")
        right.rowconfigure(1, weight=3)
        right.rowconfigure(3, weight=2)
        right.rowconfigure(5, weight=2)
        right.columnconfigure(0, weight=1)

        ttk.Label(right, text="Valid placements").grid(row=0, column=0, sticky="w")

        self.listbox = tk.Listbox(right, exportselection=False, font=("Consolas", 10))
        self.listbox.grid(row=1, column=0, sticky="nsew")
        self.listbox.bind("<<ListboxSelect>>", self.on_listbox_select)

        ttk.Label(right, text="Selected placement info").grid(row=2, column=0, sticky="w", pady=(8, 0))
        self.info_text = ScrolledText(right, height=10, wrap=tk.WORD, font=("Consolas", 10))
        self.info_text.grid(row=3, column=0, sticky="nsew")

        ttk.Label(right, text="Illustrative replay scripts").grid(row=4, column=0, sticky="w", pady=(8, 0))
        self.replay_text = ScrolledText(right, height=10, wrap=tk.WORD, font=("Consolas", 10))
        self.replay_text.grid(row=5, column=0, sticky="nsew")

    def refresh_templates(self) -> None:
        self.template_choices = list_board_templates()
        self.template_box["values"] = self.template_choices

    def load_selected_template(self) -> None:
        try:
            self.board = load_board_template(self.template_var.get())
        except Exception as exc:
            messagebox.showerror("Board load error", str(exc))
            self.board = np.zeros((BOARD_HEIGHT, BOARD_WIDTH), dtype=np.uint8)
        self.recompute_only()

    def clear_board(self) -> None:
        self.board = np.zeros((BOARD_HEIGHT, BOARD_WIDTH), dtype=np.uint8)
        self.recompute_only()

    def save_current_board(self) -> None:
        stem = simpledialog.askstring("Save board", "Board file name (without .txt):", parent=self.root)
        if stem is None:
            return
        try:
            path = save_board_template(stem, self.board)
            self.refresh_templates()
            self.template_var.set(path.stem)
            self.update_status(extra=f"saved={path.name}")
        except Exception as exc:
            messagebox.showerror("Save error", str(exc))

    def recompute_only(self) -> None:
        self.mask, self.candidates, self.outcomes = enumerate_candidates(self.board, self.piece_var.get())

        current_ids = [c.action_id for c in self.candidates]
        if self.selected_action_id not in current_ids:
            self.selected_action_id = current_ids[0] if current_ids else None

        self.populate_listbox()
        self.update_status()
        self.render()

    def populate_listbox(self) -> None:
        self.listbox.delete(0, tk.END)
        for outcome in self.outcomes:
            c = outcome.candidate
            f = outcome.features
            line = (
                f"id={c.action_id:2d}  "
                f"ori={c.orientation_id}  "
                f"x={c.x}  y={c.y_final:2d}  "
                f"lines={outcome.lines_cleared}  "
                f"holes={f.holes:2d}  "
                f"agg={f.aggregate_height:3d}  "
                f"bump={f.bumpiness:2d}"
            )
            self.listbox.insert(tk.END, line)

        if self.selected_action_id is not None:
            for idx, candidate in enumerate(self.candidates):
                if candidate.action_id == self.selected_action_id:
                    self.listbox.selection_clear(0, tk.END)
                    self.listbox.selection_set(idx)
                    self.listbox.see(idx)
                    break

    def update_status(self, extra: str | None = None) -> None:
        piece_id = self.piece_var.get()
        spawn_overlap = spawn_overlaps_visible_stack(self.board, piece_id)
        has_valid = has_any_valid_action(self.board, piece_id)

        status = (
            f"board={self.template_var.get()} | "
            f"valid_actions={len(self.candidates)} / {ACTION_SPACE_SIZE} | "
            f"spawn_overlap_visible≈{spawn_overlap} | "
            f"has_valid_action={has_valid} | "
            f"edit: left-click toggles cell, right-click clears"
        )
        if extra:
            status += f" | {extra}"
        self.status_label.config(text=status)

    def on_listbox_select(self, event=None) -> None:
        selection = self.listbox.curselection()
        if not selection:
            return
        idx = selection[0]
        if 0 <= idx < len(self.candidates):
            self.selected_action_id = self.candidates[idx].action_id
            self.render()

    def on_canvas_click(self, event) -> None:
        if event.inaxes != self.ax_board:
            return
        if event.xdata is None or event.ydata is None:
            return

        x = int(round(event.xdata))
        y = int(round(event.ydata))

        if not (0 <= x < BOARD_WIDTH and 0 <= y < BOARD_HEIGHT):
            return

        if event.button == 3:
            self.board[y, x] = 0
        else:
            self.board[y, x] = 0 if self.board[y, x] else 1

        self.recompute_only()

    def render(self) -> None:
        selected_candidate = candidate_by_action_id(self.candidates, self.selected_action_id) if self.selected_action_id is not None else None
        selected_outcome = outcome_by_action_id(self.outcomes, self.selected_action_id) if self.selected_action_id is not None else None

        self.ax_board.clear()
        self.ax_encoding.clear()
        self.ax_result.clear()

        self._draw_board_panel(selected_candidate)
        self._draw_encoding_panel(selected_candidate, selected_outcome)
        self._draw_result_panel(selected_outcome)

        self.fig.tight_layout()
        self.canvas.draw()

        self._update_text_panels(selected_candidate, selected_outcome)

    def _draw_board_panel(self, selected_candidate) -> None:
        board_img = occupancy_encoding(self.board)
        heat = candidates_heatmap(self.candidates)
        selected_mask = selected_candidate_mask(selected_candidate)

        self.ax_board.imshow(board_img, cmap="Greys", vmin=0, vmax=1, origin="upper")
        if np.max(heat) > 0:
            self.ax_board.imshow(heat, cmap="viridis", alpha=0.55, origin="upper")
        if selected_candidate is not None:
            self.ax_board.imshow(selected_mask, cmap="autumn", alpha=0.70, origin="upper")

        self.ax_board.set_title("Editable board + valid placements")
        self._style_axis(self.ax_board)

        piece_id = self.piece_var.get()
        spawn_ori = default_spawn_orientation(piece_id)
        spawn_x = default_spawn_x(piece_id)
        spawn_y = default_spawn_y(piece_id)
        for dx, dy in spawn_ori.cells_local:
            x = spawn_x + dx
            y = spawn_y + dy
            if 0 <= y < BOARD_HEIGHT:
                self.ax_board.add_patch(
                    patches.Rectangle(
                        (x - 0.5, y - 0.5),
                        1,
                        1,
                        fill=False,
                        linestyle="--",
                        linewidth=1.2,
                    )
                )

    def _draw_encoding_panel(self, selected_candidate, selected_outcome) -> None:
        mode = self.encoding_var.get()

        if mode == "occupancy":
            img = occupancy_encoding(self.board)
            title = "Occupancy encoding"
        elif mode == "heights":
            img = heights_encoding(self.board)
            title = "Height encoding"
        elif mode == "holes":
            img = hole_mask(self.board).astype(np.float32)
            title = "Hole encoding"
        elif mode == "candidates":
            img = candidates_heatmap(self.candidates)
            title = "Candidate heatmap"
        elif mode == "result":
            img = occupancy_encoding(selected_outcome.board_after) if selected_outcome is not None else occupancy_encoding(self.board)
            title = "Selected result encoding"
        else:
            img = occupancy_encoding(self.board)
            title = mode

        self.ax_encoding.imshow(img, cmap="viridis", origin="upper")
        if selected_candidate is not None and mode != "result":
            self.ax_encoding.imshow(selected_candidate_mask(selected_candidate), cmap="autumn", alpha=0.60, origin="upper")

        self.ax_encoding.set_title(title)
        self._style_axis(self.ax_encoding)

    def _draw_result_panel(self, selected_outcome) -> None:
        img = occupancy_encoding(self.board) if selected_outcome is None else occupancy_encoding(selected_outcome.board_after)
        self.ax_result.imshow(img, cmap="Greys", vmin=0, vmax=1, origin="upper")
        self.ax_result.set_title("Board after selected placement")
        self._style_axis(self.ax_result)

    def _style_axis(self, ax) -> None:
        ax.set_xticks(range(BOARD_WIDTH))
        ax.set_yticks(range(BOARD_HEIGHT))
        ax.set_xlim(-0.5, BOARD_WIDTH - 0.5)
        ax.set_ylim(BOARD_HEIGHT - 0.5, -0.5)
        ax.grid(True, linewidth=0.3)

    def _update_text_panels(self, selected_candidate, selected_outcome) -> None:
        self.info_text.delete("1.0", tk.END)
        self.replay_text.delete("1.0", tk.END)

        if selected_candidate is None or selected_outcome is None:
            self.info_text.insert(tk.END, "No selected placement.\n")
            self.replay_text.insert(tk.END, "No replay script available.\n")
            return

        c = selected_candidate
        f = selected_outcome.features

        info = (
            f"action_id:        {c.action_id}\n"
            f"piece:            {c.piece_id}\n"
            f"orientation_id:   {c.orientation_id}\n"
            f"x:                {c.x}\n"
            f"y_final:          {c.y_final}\n"
            f"cells_world:      {c.cells_world}\n\n"
            f"lines_cleared:    {selected_outcome.lines_cleared}\n"
            f"aggregate_height: {f.aggregate_height}\n"
            f"holes:            {f.holes}\n"
            f"bumpiness:        {f.bumpiness}\n"
            f"max_height:       {f.max_height}\n"
        )
        self.info_text.insert(tk.END, info)

        scripts = build_replay_scripts(self.piece_var.get(), selected_candidate)
        replay_text = (
            "These are illustrative replay styles for visualization.\n"
            "They are not controller-exact path proofs.\n\n"
        )
        for name, script in scripts.items():
            replay_text += f"{name}\n  {script}\n\n"
        self.replay_text.insert(tk.END, replay_text)


def main() -> None:
    args = parse_args()
    root = tk.Tk()
    PlacementTesterApp(
        root=root,
        initial_preset=args.preset,
        initial_piece=args.piece,
        initial_encoding=args.encoding,
    )
    root.mainloop()


if __name__ == "__main__":
    main()