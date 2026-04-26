from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[0]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import matplotlib.pyplot as plt
import numpy as np

from src.envs.placement_core import enumerate_candidates
from src.envs.placement_env import PlacementTetrisEnv, RewardConfig


RUNS_DIR = PROJECT_ROOT / "runs" / "heuristic_eval"


@dataclass(frozen=True)
class HeuristicWeights:
    single_score: float = 2.0
    double_score: float = 8.0
    triple_score: float = 32.0
    tetris_score: float = 120.0
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


@dataclass
class EpisodeStats:
    episode: int
    seed: int
    total_reward: float
    steps: int
    total_lines: int
    total_score: float
    singles: int
    doubles: int
    triples: int
    tetrises: int
    invalid_action: bool
    terminated: bool
    truncated: bool
    duration_sec: float
    reward_per_step: float
    lines_per_step: float


DEFAULT_WEIGHTS = HeuristicWeights()
FIELD_NAMES = {f.name for f in fields(HeuristicWeights)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a heuristic placement agent on PlacementTetrisEnv with verbose logging.")
    parser.add_argument("--episodes", type=int, default=25, help="Number of episodes to run.")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed.")
    parser.add_argument("--max-steps", type=int, default=500, help="Episode truncation limit inside the env.")
    parser.add_argument("--preset", type=str, default=None, help="Optional preset board name.")
    parser.add_argument("--render", action="store_true", help="Show visual Tetris window (OpenCV).")
    parser.add_argument("--delay-ms", type=int, default=120, help="Delay between frames when rendering.")
    parser.add_argument("--upscale", type=int, default=30, help="Render upscale factor.")
    parser.add_argument("--print-board", action="store_true", help="Print ANSI board after each step.")
    parser.add_argument("--plot", action="store_true", help="Save summary plots.")
    parser.add_argument("--save-csv", action="store_true", help="Save per-episode metrics CSV.")
    parser.add_argument("--save-json", action="store_true", help="Save summary JSON.")
    parser.add_argument("--save-prefix", type=str, default="heuristic_agent", help="Prefix for saved output filenames.")
    parser.add_argument("--weights-json", type=str, default=None, help="Path to a JSON config or best_weights/best_model file.")
    parser.add_argument("--lookahead-depth", type=int, default=0, choices=[0, 1], help="0 = greedy, 1 = one-piece lookahead.")
    parser.add_argument("--lookahead-weight", type=float, default=None, help="Override lookahead weight; defaults to value from JSON if present, otherwise 0.6.")
    parser.add_argument("--progress-every", type=int, default=25, help="Print progress every N steps within each episode. Use 0 to disable step-based progress.")
    parser.add_argument("--progress-seconds", type=float, default=0.0, help="Also print progress whenever this many seconds have passed since the last progress line.")

    parser.add_argument("--w-single", type=float, default=None, help="Bonus for single line.")
    parser.add_argument("--w-double", type=float, default=None, help="Bonus for double line.")
    parser.add_argument("--w-triple", type=float, default=None, help="Bonus for triple line.")
    parser.add_argument("--w-tetris", type=float, default=None, help="Bonus for tetris (4 lines).")
    parser.add_argument("--w-height", type=float, default=None, help="Penalty weight for aggregate height.")
    parser.add_argument("--w-holes", type=float, default=None, help="Penalty weight for holes.")
    parser.add_argument("--w-bump", type=float, default=None, help="Penalty weight for bumpiness.")
    parser.add_argument("--w-max", type=float, default=None, help="Penalty weight for max height.")
    parser.add_argument("--w-row-trans", type=float, default=None, help="Penalty weight for row transitions.")
    parser.add_argument("--w-col-trans", type=float, default=None, help="Penalty weight for column transitions.")
    parser.add_argument("--w-rows-holes", type=float, default=None, help="Penalty weight for rows with holes.")
    parser.add_argument("--w-hole-depth", type=float, default=None, help="Penalty weight for hole depth.")
    parser.add_argument("--w-wells", type=float, default=None, help="Penalty weight for cumulative wells.")
    parser.add_argument("--w-eroded", type=float, default=None, help="Bonus weight for eroded piece cells.")

    return parser.parse_args()


def maybe_show_frame(frame: np.ndarray | None, delay_ms: int) -> bool:
    if frame is None:
        return False
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    cv2.imshow("Placement Tetris - Heuristic Agent", frame_bgr)
    key = cv2.waitKey(delay_ms) & 0xFF
    return key == ord("q")


def plot_series(values: np.ndarray, ylabel: str, title: str, save_path: Path) -> None:
    plt.figure(figsize=(8, 4.5))
    plt.plot(np.arange(1, len(values) + 1), values)
    plt.xlabel("Episode")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def maybe_save_plots(stats: list[EpisodeStats], prefix: str) -> None:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    rewards = np.array([s.total_reward for s in stats], dtype=np.float32)
    lengths = np.array([s.steps for s in stats], dtype=np.int32)
    total_lines = np.array([s.total_lines for s in stats], dtype=np.int32)
    total_scores = np.array([s.total_score for s in stats], dtype=np.float32)
    singles = np.array([s.singles for s in stats], dtype=np.int32)
    doubles = np.array([s.doubles for s in stats], dtype=np.int32)
    triples = np.array([s.triples for s in stats], dtype=np.int32)
    tetrises = np.array([s.tetrises for s in stats], dtype=np.int32)

    plot_series(rewards, "Episode Reward", "Heuristic Agent: Episode Reward", RUNS_DIR / f"{prefix}_rewards.png")
    plot_series(lengths, "Episode Length", "Heuristic Agent: Episode Length", RUNS_DIR / f"{prefix}_lengths.png")
    plot_series(total_lines, "Total Lines Cleared", "Heuristic Agent: Total Lines Cleared", RUNS_DIR / f"{prefix}_lines.png")
    plot_series(total_scores, "Total Score", "Heuristic Agent: Total Score", RUNS_DIR / f"{prefix}_score.png")
    plot_series(singles, "Singles", "Heuristic Agent: Singles", RUNS_DIR / f"{prefix}_singles.png")
    plot_series(doubles, "Doubles", "Heuristic Agent: Doubles", RUNS_DIR / f"{prefix}_doubles.png")
    plot_series(triples, "Triples", "Heuristic Agent: Triples", RUNS_DIR / f"{prefix}_triples.png")
    plot_series(tetrises, "Tetrises", "Heuristic Agent: Tetrises", RUNS_DIR / f"{prefix}_tetrises.png")

    print(f"\nSaved plots to: {RUNS_DIR.resolve()}")


def save_episode_csv(stats: list[EpisodeStats], prefix: str) -> Path:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    path = RUNS_DIR / f"{prefix}_episodes.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(stats[0]).keys()))
        writer.writeheader()
        for row in stats:
            writer.writerow(asdict(row))
    return path


def save_summary_json(stats: list[EpisodeStats], weights: HeuristicWeights, lookahead_depth: int, lookahead_weight: float, prefix: str) -> Path:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    rewards = np.array([s.total_reward for s in stats], dtype=np.float32)
    steps = np.array([s.steps for s in stats], dtype=np.int32)
    total_lines = np.array([s.total_lines for s in stats], dtype=np.int32)
    total_score = np.array([s.total_score for s in stats], dtype=np.float32)
    singles = np.array([s.singles for s in stats], dtype=np.int32)
    doubles = np.array([s.doubles for s in stats], dtype=np.int32)
    triples = np.array([s.triples for s in stats], dtype=np.int32)
    tetrises = np.array([s.tetrises for s in stats], dtype=np.int32)
    payload = {
        "lookahead_depth": lookahead_depth,
        "lookahead_weight": lookahead_weight,
        "weights": asdict(weights),
        "episodes": [asdict(s) for s in stats],
        "summary": {
            "episode_count": len(stats),
            "mean_reward": float(rewards.mean()),
            "std_reward": float(rewards.std()),
            "mean_steps": float(steps.mean()),
            "mean_total_lines": float(total_lines.mean()),
            "mean_total_score": float(total_score.mean()),
            "mean_singles": float(singles.mean()),
            "mean_doubles": float(doubles.mean()),
            "mean_triples": float(triples.mean()),
            "mean_tetrises": float(tetrises.mean()),
            "top_out_rate": float(np.mean([int(s.terminated and not s.truncated) for s in stats])),
        },
    }
    path = RUNS_DIR / f"{prefix}_summary.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return path


def load_json_payload(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        for key in ("best_individual", "individual", "weights"):
            value = data.get(key)
            if isinstance(value, dict):
                return value
        return data
    raise ValueError(f"Expected a JSON object in {path}, got {type(data).__name__}")


def build_weights(args: argparse.Namespace) -> tuple[HeuristicWeights, float]:
    payload: dict[str, Any] = {}
    if args.weights_json is not None:
        payload = load_json_payload(args.weights_json)

    merged = {name: getattr(DEFAULT_WEIGHTS, name) for name in FIELD_NAMES}
    for name in FIELD_NAMES:
        if name in payload and payload[name] is not None:
            merged[name] = float(payload[name])

    cli_overrides = {
        "single_score": args.w_single,
        "double_score": args.w_double,
        "triple_score": args.w_triple,
        "tetris_score": args.w_tetris,
        "aggregate_height_weight": args.w_height,
        "holes_weight": args.w_holes,
        "bumpiness_weight": args.w_bump,
        "max_height_weight": args.w_max,
        "row_transitions_weight": args.w_row_trans,
        "column_transitions_weight": args.w_col_trans,
        "rows_with_holes_weight": args.w_rows_holes,
        "hole_depth_weight": args.w_hole_depth,
        "cumulative_wells_weight": args.w_wells,
        "eroded_piece_cells_weight": args.w_eroded,
    }
    for key, value in cli_overrides.items():
        if value is not None:
            merged[key] = float(value)

    lookahead_weight = float(args.lookahead_weight) if args.lookahead_weight is not None else float(payload.get("lookahead_weight", 0.6))
    return HeuristicWeights(**merged), lookahead_weight


def board_value(features, weights: HeuristicWeights) -> float:
    return -(
        weights.aggregate_height_weight * float(features.aggregate_height)
        + weights.holes_weight * float(features.holes)
        + weights.bumpiness_weight * float(features.bumpiness)
        + weights.max_height_weight * float(features.max_height)
        + weights.row_transitions_weight * float(getattr(features, "row_transitions", 0.0))
        + weights.column_transitions_weight * float(getattr(features, "column_transitions", 0.0))
        + weights.rows_with_holes_weight * float(getattr(features, "rows_with_holes", 0.0))
        + weights.hole_depth_weight * float(getattr(features, "hole_depth", 0.0))
        + weights.cumulative_wells_weight * float(getattr(features, "cumulative_wells", 0.0))
    )


def score_outcome(outcome, weights: HeuristicWeights) -> float:
    lines = int(outcome.lines_cleared)
    if lines == 1:
        score_delta = weights.single_score
    elif lines == 2:
        score_delta = weights.double_score
    elif lines == 3:
        score_delta = weights.triple_score
    elif lines >= 4:
        score_delta = weights.tetris_score
    else:
        score_delta = 0.0
    eroded = float(getattr(outcome, "eroded_piece_cells", 0.0))
    return score_delta + weights.eroded_piece_cells_weight * eroded + board_value(outcome.features, weights)


def choose_best_action_no_lookahead(env: PlacementTetrisEnv, weights: HeuristicWeights) -> tuple[int, float]:
    _, _, outcomes = enumerate_candidates(env.board, env.current_piece)
    if not outcomes:
        raise RuntimeError("No valid outcomes available when choosing best action.")

    best_action = None
    best_score = -float("inf")
    best_tiebreak = None
    for outcome in outcomes:
        score = score_outcome(outcome, weights)
        tie = (int(outcome.lines_cleared), -int(outcome.features.holes), -int(outcome.features.aggregate_height))
        if score > best_score or (score == best_score and (best_tiebreak is None or tie > best_tiebreak)):
            best_score = score
            best_action = int(outcome.candidate.action_id)
            best_tiebreak = tie
    if best_action is None:
        raise RuntimeError("Failed to choose a best action.")
    return best_action, best_score


def choose_best_action_with_lookahead(env: PlacementTetrisEnv, weights: HeuristicWeights, lookahead_weight: float) -> tuple[int, float]:
    _, _, first_outcomes = enumerate_candidates(env.board, env.current_piece)
    if not first_outcomes:
        raise RuntimeError("No valid outcomes available when choosing best action.")

    best_action = None
    best_total = -float("inf")
    best_tiebreak = None
    for first in first_outcomes:
        immediate = score_outcome(first, weights)
        _, _, second_outcomes = enumerate_candidates(first.board_after, env.next_piece)
        future = max(score_outcome(second, weights) for second in second_outcomes) if second_outcomes else -1e9
        total = immediate + lookahead_weight * future
        tie = (int(first.lines_cleared), -int(first.features.holes), -int(first.features.aggregate_height))
        if total > best_total or (total == best_total and (best_tiebreak is None or tie > best_tiebreak)):
            best_total = total
            best_action = int(first.candidate.action_id)
            best_tiebreak = tie
    if best_action is None:
        raise RuntimeError("Failed to choose a best action.")
    return best_action, best_total


def choose_action(env: PlacementTetrisEnv, weights: HeuristicWeights, lookahead_depth: int, lookahead_weight: float) -> tuple[int, float]:
    return choose_best_action_with_lookahead(env, weights, lookahead_weight) if lookahead_depth > 0 else choose_best_action_no_lookahead(env, weights)


def format_progress(ep: int, steps: int, total_reward: float, info: dict[str, Any], singles: int, doubles: int, triples: int, tetrises: int, heuristic_score: float) -> str:
    return (
        f"Ep {ep:02d} | step={steps:4d} | reward={total_reward:9.2f} | score={float(info.get('total_score', 0.0)):9.2f} | "
        f"lines={int(info.get('total_lines_cleared', 0)):4d} | S/D/T/T4={singles}/{doubles}/{triples}/{tetrises} | "
        f"cur={info.get('current_piece')} next={info.get('next_piece')} | h={heuristic_score:9.2f}"
    )


def main() -> None:
    args = parse_args()
    weights, lookahead_weight = build_weights(args)

    if args.weights_json is not None:
        print(f"Loaded weight config from: {Path(args.weights_json).resolve()}")
    print(f"Using lookahead depth={args.lookahead_depth}, lookahead_weight={lookahead_weight:.3f}")
    print(f"Progress logging: every {args.progress_every} steps" + (f" and every {args.progress_seconds:.1f}s" if args.progress_seconds > 0 else ""))

    reward_config = RewardConfig(
        single_score=weights.single_score,
        double_score=weights.double_score,
        triple_score=weights.triple_score,
        tetris_score=weights.tetris_score,
        shaping_scale=0.02,
        shaping_gamma=0.99,
        aggregate_height_weight=weights.aggregate_height_weight,
        holes_weight=weights.holes_weight,
        bumpiness_weight=weights.bumpiness_weight,
        max_height_weight=weights.max_height_weight,
        row_transitions_weight=weights.row_transitions_weight,
        column_transitions_weight=weights.column_transitions_weight,
        rows_with_holes_weight=weights.rows_with_holes_weight,
        hole_depth_weight=weights.hole_depth_weight,
        cumulative_wells_weight=weights.cumulative_wells_weight,
        eroded_piece_cells_weight=weights.eroded_piece_cells_weight,
        survival_bonus=0.0,
        game_over_penalty=-25.0,
        invalid_action_penalty=-5.0,
    )

    render_mode = "rgb_array" if args.render else "ansi" if args.print_board else None
    env_kwargs = {
        "max_steps": args.max_steps,
        "render_mode": render_mode,
        "render_upscale": args.upscale,
        "reward_config": reward_config,
    }
    if args.preset is not None:
        env_kwargs["initial_board_preset"] = args.preset
    env = PlacementTetrisEnv(**env_kwargs)

    episode_stats: list[EpisodeStats] = []

    try:
        for ep in range(args.episodes):
            episode_seed = args.seed + ep
            _, info = env.reset(seed=episode_seed)
            terminated = False
            truncated = False
            total_reward = 0.0
            steps = 0
            singles = doubles = triples = tetrises = 0
            start_time = time.time()
            last_progress_time = start_time

            print(
                f"\nEpisode {ep + 1:02d} start | seed={episode_seed} | "
                f"current={info['current_piece']} next={info['next_piece']} | valid_actions={info['valid_action_count']}"
            )

            if args.render:
                frame = env.render()
                if maybe_show_frame(frame, args.delay_ms):
                    raise KeyboardInterrupt("Quit requested by user.")

            while not (terminated or truncated):
                if not env.has_valid_actions():
                    print("No valid actions available.")
                    break

                action, heuristic_score = choose_action(env, weights, args.lookahead_depth, lookahead_weight)
                _, reward, terminated, truncated, info = env.step(action)
                total_reward += float(reward)
                steps += 1

                cleared = int(info.get("lines_cleared_this_step", 0))
                if cleared == 1:
                    singles += 1
                elif cleared == 2:
                    doubles += 1
                elif cleared == 3:
                    triples += 1
                elif cleared >= 4:
                    tetrises += 1

                now = time.time()
                should_log = False
                if args.progress_every > 0 and steps % args.progress_every == 0:
                    should_log = True
                if args.progress_seconds > 0 and (now - last_progress_time) >= args.progress_seconds:
                    should_log = True
                if should_log:
                    print(format_progress(ep + 1, steps, total_reward, info, singles, doubles, triples, tetrises, heuristic_score))
                    last_progress_time = now

                if args.render:
                    frame = env.render()
                    if maybe_show_frame(frame, args.delay_ms):
                        raise KeyboardInterrupt("Quit requested by user.")

                if args.print_board:
                    ansi = env.render()
                    if ansi is not None:
                        print(ansi)
                        print(f"heuristic_score={heuristic_score:.3f}")
                        print("-" * 40)

            duration = time.time() - start_time
            ep_stats = EpisodeStats(
                episode=ep + 1,
                seed=episode_seed,
                total_reward=float(total_reward),
                steps=int(steps),
                total_lines=int(info.get("total_lines_cleared", 0)),
                total_score=float(info.get("total_score", 0.0)),
                singles=int(singles),
                doubles=int(doubles),
                triples=int(triples),
                tetrises=int(tetrises),
                invalid_action=bool(info.get("invalid_action", False)),
                terminated=bool(terminated),
                truncated=bool(truncated),
                duration_sec=float(duration),
                reward_per_step=float(total_reward / max(steps, 1)),
                lines_per_step=float(info.get("total_lines_cleared", 0) / max(steps, 1)),
            )
            episode_stats.append(ep_stats)

            print(
                f"Episode {ep + 1:02d} end   | reward={ep_stats.total_reward:9.3f} | steps={ep_stats.steps:4d} | "
                f"score={ep_stats.total_score:9.2f} | lines={ep_stats.total_lines:4d} | "
                f"S/D/T/T4={ep_stats.singles}/{ep_stats.doubles}/{ep_stats.triples}/{ep_stats.tetrises} | "
                f"r/step={ep_stats.reward_per_step:.4f} | terminated={terminated} truncated={truncated} | t={duration:.2f}s"
            )

    except KeyboardInterrupt as exc:
        print(f"\nStopped early: {exc}")

    finally:
        env.close()
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass

    if not episode_stats:
        print("\nNo completed episodes to summarize.")
        return

    rewards = np.array([s.total_reward for s in episode_stats], dtype=np.float32)
    lengths = np.array([s.steps for s in episode_stats], dtype=np.int32)
    total_lines = np.array([s.total_lines for s in episode_stats], dtype=np.int32)
    total_scores = np.array([s.total_score for s in episode_stats], dtype=np.float32)
    singles = np.array([s.singles for s in episode_stats], dtype=np.int32)
    doubles = np.array([s.doubles for s in episode_stats], dtype=np.int32)
    triples = np.array([s.triples for s in episode_stats], dtype=np.int32)
    tetrises = np.array([s.tetrises for s in episode_stats], dtype=np.int32)
    invalids = np.array([int(s.invalid_action) for s in episode_stats], dtype=np.int32)
    topouts = np.array([int(s.terminated and not s.truncated) for s in episode_stats], dtype=np.int32)

    print("\nSummary")
    print(f"Episodes:            {len(rewards)}")
    print(f"Mean reward:         {rewards.mean():.3f}")
    print(f"Std reward:          {rewards.std():.3f}")
    print(f"Max reward:          {rewards.max():.3f}")
    print(f"Mean reward / step:  {(rewards.sum() / max(lengths.sum(), 1)):.4f}")
    print(f"Mean episode length: {lengths.mean():.2f}")
    print(f"Max episode length:  {lengths.max()}")
    print(f"Mean total score:    {total_scores.mean():.2f}")
    print(f"Max total score:     {total_scores.max():.2f}")
    print(f"Mean total lines:    {total_lines.mean():.2f}")
    print(f"Max total lines:     {total_lines.max()}")
    print(f"Mean singles:        {singles.mean():.2f}")
    print(f"Mean doubles:        {doubles.mean():.2f}")
    print(f"Mean triples:        {triples.mean():.2f}")
    print(f"Mean tetrises:       {tetrises.mean():.2f}")
    print(f"Top-out rate:        {topouts.mean():.2%}")
    print(f"Invalid episodes:    {invalids.sum()}")

    if args.plot:
        maybe_save_plots(episode_stats, args.save_prefix)
    if args.save_csv:
        path = save_episode_csv(episode_stats, args.save_prefix)
        print(f"Saved episode CSV to: {path.resolve()}")
    if args.save_json:
        path = save_summary_json(episode_stats, weights, args.lookahead_depth, lookahead_weight, args.save_prefix)
        print(f"Saved summary JSON to: {path.resolve()}")


if __name__ == "__main__":
    main()
