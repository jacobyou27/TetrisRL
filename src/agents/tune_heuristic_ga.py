
from __future__ import annotations

import argparse
import concurrent.futures as cf
import csv
import json
import logging
import math
import multiprocessing as mp
import random
import sys
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agents.heuristic_core import (
    HeuristicWeights,
    choose_best_action_no_lookahead,
    choose_best_action_with_lookahead,
)

# Visualization helpers are imported lazily so worker processes do not pay GUI import costs.
from src.envs.placement_env import PlacementTetrisEnv, RewardConfig

RUNS_DIR = PROJECT_ROOT / "runs" / "ga"


def _viz_modules():
    from src.agents.ga_visualization import (
        LiveDashboard,
        ShowcaseVideoWriter,
        ShowcaseWindow,
        compose_panels,
        draw_board_panel,
    )
    return LiveDashboard, ShowcaseVideoWriter, ShowcaseWindow, compose_panels, draw_board_panel


def setup_logging(log_dir: Path, verbosity: int = 1) -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "ga_run.log"
    logger = logging.getLogger(f"ga.{log_dir.name}")
    logger.handlers.clear()
    logger.setLevel(logging.DEBUG)

    fh = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)-8s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
    logger.addHandler(fh)

    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG if verbosity >= 2 else logging.INFO if verbosity == 1 else logging.WARNING)
    ch.setFormatter(logging.Formatter("%(message)s" if verbosity < 2 else "%(asctime)s [%(levelname)-8s] %(message)s", datefmt="%H:%M:%S"))
    logger.addHandler(ch)
    return logger


def save_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def plot_fitness_evolution(history: list[dict], output_dir: Path, logger: logging.Logger) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available; skipping fitness plot")
        return
    generations = [h["generation"] + 1 for h in history]
    best_fitness = [h["best_fitness"] for h in history]
    mean_fitness = [h["mean_fitness"] for h in history]
    std_fitness = [h["std_fitness"] for h in history]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(generations, best_fitness, "g-o", label="Best Fitness", linewidth=2)
    ax.plot(generations, mean_fitness, "b-s", label="Mean Fitness", linewidth=2)
    ax.fill_between(generations, np.array(mean_fitness) - np.array(std_fitness), np.array(mean_fitness) + np.array(std_fitness), alpha=0.2, label="Std Dev")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Fitness")
    ax.set_title("GA Fitness Evolution")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plot_path = output_dir / "fitness_evolution.png"
    fig.savefig(plot_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved fitness plot to {plot_path.name}")


def plot_metrics_evolution(history: list[dict], output_dir: Path, logger: logging.Logger) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    generations = [h["generation"] + 1 for h in history]
    lines = [h["best_mean_lines"] for h in history]
    scores = [h["best_mean_score"] for h in history]
    topout = [h["best_top_out_rate"] for h in history]
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 10))
    ax1.plot(generations, lines, "b-o", linewidth=2)
    ax1.set_ylabel("Mean Lines Cleared")
    ax1.grid(True, alpha=0.3)
    ax2.plot(generations, scores, "g-s", linewidth=2)
    ax2.set_ylabel("Mean Score")
    ax2.grid(True, alpha=0.3)
    ax3.plot(generations, topout, "r-^", linewidth=2)
    ax3.set_ylabel("Top-Out Rate")
    ax3.set_xlabel("Generation")
    ax3.grid(True, alpha=0.3)
    fig.tight_layout()
    plot_path = output_dir / "metrics_evolution.png"
    fig.savefig(plot_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved metrics plot to {plot_path.name}")


def plot_clear_mix_evolution(history: list[dict], output_dir: Path, logger: logging.Logger) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    generations = [h["generation"] + 1 for h in history]
    singles = [h.get("best_mean_singles", 0.0) for h in history]
    doubles = [h.get("best_mean_doubles", 0.0) for h in history]
    triples = [h.get("best_mean_triples", 0.0) for h in history]
    tetrises = [h.get("best_mean_tetrises", 0.0) for h in history]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(generations, singles, label="Singles", linewidth=2)
    ax.plot(generations, doubles, label="Doubles", linewidth=2)
    ax.plot(generations, triples, label="Triples", linewidth=2)
    ax.plot(generations, tetrises, label="Tetrises", linewidth=2)
    ax.set_xlabel("Generation")
    ax.set_ylabel("Average clears per evaluation")
    ax.set_title("Best Individual Clear Mix")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plot_path = output_dir / "clear_mix_evolution.png"
    fig.savefig(plot_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved clear-mix plot to {plot_path.name}")


def plot_final_weights(best_individual_dict: dict[str, float], output_dir: Path, logger: logging.Logger) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    score_keys = ["single_score", "double_score", "triple_score", "tetris_score"]
    feature_keys = [k for k in best_individual_dict.keys() if k.endswith("_weight")]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    score_vals = [best_individual_dict[k] for k in score_keys]
    ax1.bar(range(len(score_keys)), score_vals)
    ax1.set_xticks(range(len(score_keys)))
    ax1.set_xticklabels([k.replace("_", "\n") for k in score_keys], fontsize=9)
    ax1.set_title("Evolved Line Clear Bonuses")
    ax1.grid(True, axis="y", alpha=0.3)
    feature_vals = [best_individual_dict[k] for k in feature_keys]
    ax2.barh(range(len(feature_keys)), feature_vals)
    ax2.set_yticks(range(len(feature_keys)))
    ax2.set_yticklabels([k.replace("_weight", "").replace("_", " ") for k in feature_keys], fontsize=9)
    ax2.set_title("Evolved Feature Weights")
    ax2.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    plot_path = output_dir / "final_weights.png"
    fig.savefig(plot_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved weights plot to {plot_path.name}")


@dataclass(frozen=True)
class GAIndividual:
    # Tetris-friendly defaults: keep singles modest, preserve doubles as the
    # safe board-stabilizing clear, and make triples/tetrises much more valuable.
    single_score: float = 3.5
    double_score: float = 18.0
    triple_score: float = 70.0
    tetris_score: float = 240.0
    # Strong anti-hole / anti-fragmentation bias to make long clears feasible.
    aggregate_height_weight: float = 1.15
    holes_weight: float = 8.50
    bumpiness_weight: float = 0.16
    max_height_weight: float = 1.25
    row_transitions_weight: float = 1.20
    column_transitions_weight: float = 1.90
    rows_with_holes_weight: float = 2.20
    hole_depth_weight: float = 2.00
    # Allow rewarding a clean well by making this weight slightly negative.
    cumulative_wells_weight: float = -0.50
    eroded_piece_cells_weight: float = 5.00
    lookahead_weight: float = 0.60

    def to_heuristic_weights(self) -> HeuristicWeights:
        return HeuristicWeights(
            single_score=self.single_score,
            double_score=self.double_score,
            triple_score=self.triple_score,
            tetris_score=self.tetris_score,
            aggregate_height_weight=self.aggregate_height_weight,
            holes_weight=self.holes_weight,
            bumpiness_weight=self.bumpiness_weight,
            max_height_weight=self.max_height_weight,
            row_transitions_weight=self.row_transitions_weight,
            column_transitions_weight=self.column_transitions_weight,
            rows_with_holes_weight=self.rows_with_holes_weight,
            hole_depth_weight=self.hole_depth_weight,
            cumulative_wells_weight=self.cumulative_wells_weight,
            eroded_piece_cells_weight=self.eroded_piece_cells_weight,
        )

    def to_reward_config(self) -> RewardConfig:
        return RewardConfig(
            single_score=self.single_score,
            double_score=self.double_score,
            triple_score=self.triple_score,
            tetris_score=self.tetris_score,
            shaping_scale=0.02,
            shaping_gamma=0.99,
            aggregate_height_weight=self.aggregate_height_weight,
            holes_weight=self.holes_weight,
            bumpiness_weight=self.bumpiness_weight,
            max_height_weight=self.max_height_weight,
            row_transitions_weight=self.row_transitions_weight,
            column_transitions_weight=self.column_transitions_weight,
            rows_with_holes_weight=self.rows_with_holes_weight,
            hole_depth_weight=self.hole_depth_weight,
            cumulative_wells_weight=self.cumulative_wells_weight,
            eroded_piece_cells_weight=self.eroded_piece_cells_weight,
            survival_bonus=0.0,
            game_over_penalty=-25.0,
            invalid_action_penalty=-5.0,
        )


GENE_RANGES: dict[str, tuple[float, float]] = {
    # Keep singles modest so the search is not rewarded for shallow, constant singles.
    "single_score": (0.0, 4.0),
    # Doubles are still useful as a stabilizing clear, but do not cap them so hard
    # that the agent stops taking healthy cleanups.
    "double_score": (12.0, 24.0),
    "triple_score": (45.0, 110.0),
    "tetris_score": (180.0, 320.0),
    # Stronger anti-hole / anti-fragmentation caps.
    "aggregate_height_weight": (0.4, 2.4),
    "holes_weight": (5.0, 18.0),
    # Keep bumpiness from dominating; one intentional well should still be possible.
    "bumpiness_weight": (0.0, 0.9),
    "max_height_weight": (0.4, 2.5),
    "row_transitions_weight": (0.2, 3.5),
    "column_transitions_weight": (0.5, 4.0),
    "rows_with_holes_weight": (0.8, 7.0),
    "hole_depth_weight": (0.3, 4.5),
    # Allow negative values so GA can reward a clean tetris well if that helps.
    "cumulative_wells_weight": (-2.0, 1.5),
    "eroded_piece_cells_weight": (1.0, 8.5),
    "lookahead_weight": (0.0, 1.2),
}
GENE_ORDER = [f.name for f in fields(GAIndividual)]

# Fitness shaping for line-clear mix. Lines still dominate, but the search is
# now much more explicitly biased toward stable doubles and especially triples/
# tetrises, while gently discouraging endless singles.
FITNESS_SCORE_SCALE = 0.015
FITNESS_REWARD_SCALE = 0.004
FITNESS_DOUBLE_BONUS = 0.80
FITNESS_TRIPLE_BONUS = 2.00
FITNESS_TETRIS_BONUS = 4.50
FITNESS_SINGLE_PENALTY = 0.08


@dataclass
class EvalStats:
    fitness: float
    mean_lines: float
    mean_score: float
    mean_reward: float
    mean_steps: float
    top_out_rate: float
    mean_singles: float
    mean_doubles: float
    mean_triples: float
    mean_tetrises: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tune Tetris heuristic weights with a genetic algorithm.")
    parser.add_argument("--population", type=int, default=40)
    parser.add_argument("--generations", type=int, default=30)
    parser.add_argument("--episodes-per-individual", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--preset", type=str, default=None)
    parser.add_argument("--elite", type=int, default=4)
    parser.add_argument("--tournament-size", type=int, default=4)
    parser.add_argument("--mutation-rate", type=float, default=0.20)
    parser.add_argument("--mutation-scale", type=float, default=0.15)
    parser.add_argument("--blend-alpha", type=float, default=0.25)
    parser.add_argument("--lookahead-depth", type=int, default=1, choices=[0, 1])
    parser.add_argument("--workers", type=int, default=1, help="Number of worker processes. Use 1 for serial.")
    parser.add_argument("--save-name", type=str, default="heuristic_ga")
    parser.add_argument("--resume-from", type=str, default=None)
    parser.add_argument("--verbosity", type=int, default=1, choices=[0, 1, 2])
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--plot-every", type=int, default=0)
    parser.add_argument("--live-dashboard", action="store_true", help="Show a live matplotlib dashboard while GA runs.")
    parser.add_argument("--live-boards", action="store_true", help="Show side-by-side boards for showcase generations.")
    parser.add_argument("--showcase-every", type=int, default=1, help="Visualize boards every N generations when --live-boards is on.")
    parser.add_argument("--showcase-steps", type=int, default=160)
    parser.add_argument("--showcase-delay-ms", type=int, default=65)
    parser.add_argument("--showcase-upscale", type=int, default=18)
    parser.add_argument("--showcase-save-frames", action="store_true", help="Save a PNG snapshot for each showcase generation.")
    parser.add_argument("--showcase-save-video", action="store_true", help="Save a showcase MP4 for each showcase generation.")
    parser.add_argument("--showcase-video-fps", type=int, default=12)
    parser.add_argument("--text-topk", type=int, default=3, help="Print the top-K individuals each generation.")
    parser.add_argument("--save-population-snapshots", action="store_true")
    parser.add_argument("--no-history-csv", action="store_true")
    return parser.parse_args()


def clamp_gene(name: str, value: float) -> float:
    lo, hi = GENE_RANGES[name]
    return float(min(max(value, lo), hi))


def individual_to_dict(individual: GAIndividual) -> dict[str, float]:
    return {name: float(getattr(individual, name)) for name in GENE_ORDER}


def dict_to_individual(data: dict[str, float]) -> GAIndividual:
    return GAIndividual(**{name: clamp_gene(name, float(data[name])) for name in GENE_ORDER})


def random_individual(rng: random.Random) -> GAIndividual:
    return GAIndividual(**{name: rng.uniform(*GENE_RANGES[name]) for name in GENE_ORDER})


def blend_crossover(a: GAIndividual, b: GAIndividual, rng: random.Random, alpha: float) -> GAIndividual:
    child = {}
    for name in GENE_ORDER:
        x = float(getattr(a, name))
        y = float(getattr(b, name))
        lo = min(x, y)
        hi = max(x, y)
        span = hi - lo
        child[name] = clamp_gene(name, rng.uniform(lo - alpha * span, hi + alpha * span))
    return GAIndividual(**child)


def mutate(individual: GAIndividual, rng: random.Random, mutation_rate: float, mutation_scale: float) -> GAIndividual:
    data = individual_to_dict(individual)
    for name in GENE_ORDER:
        if rng.random() > mutation_rate:
            continue
        lo, hi = GENE_RANGES[name]
        sigma = max((hi - lo) * mutation_scale, 1e-6)
        data[name] = clamp_gene(name, data[name] + rng.gauss(0.0, sigma))
    return GAIndividual(**data)


def tournament_select(population: list[GAIndividual], fitnesses: list[float], rng: random.Random, tournament_size: int) -> GAIndividual:
    indices = [rng.randrange(len(population)) for _ in range(tournament_size)]
    best_idx = max(indices, key=lambda idx: fitnesses[idx])
    return population[best_idx]


def choose_action(env: PlacementTetrisEnv, weights: HeuristicWeights, lookahead_depth: int, lookahead_weight: float) -> tuple[int, float]:
    if lookahead_depth > 0:
        return choose_best_action_with_lookahead(env, weights, lookahead_weight)
    return choose_best_action_no_lookahead(env, weights)


def evaluate_individual(individual: GAIndividual, *, episodes: int, base_seed: int, max_steps: int, preset: str | None, lookahead_depth: int) -> EvalStats:
    env = PlacementTetrisEnv(reward_config=individual.to_reward_config(), max_steps=max_steps, initial_board_preset=preset)
    weights = individual.to_heuristic_weights()
    rewards: list[float] = []
    lines: list[int] = []
    scores: list[float] = []
    steps_list: list[int] = []
    singles_list: list[int] = []
    doubles_list: list[int] = []
    triples_list: list[int] = []
    tetrises_list: list[int] = []
    top_outs = 0
    try:
        for ep in range(episodes):
            _, info = env.reset(seed=base_seed + ep)
            terminated = False
            truncated = False
            total_reward = 0.0
            steps = 0
            singles = 0
            doubles = 0
            triples = 0
            tetrises = 0
            while not (terminated or truncated):
                if not env.has_valid_actions():
                    break
                action, _ = choose_action(env, weights, lookahead_depth, individual.lookahead_weight)
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
            rewards.append(total_reward)
            lines.append(int(info.get("total_lines_cleared", 0)))
            scores.append(float(info.get("total_score", 0.0)))
            steps_list.append(steps)
            singles_list.append(singles)
            doubles_list.append(doubles)
            triples_list.append(triples)
            tetrises_list.append(tetrises)
            if terminated and not truncated:
                top_outs += 1
    finally:
        env.close()
    mean_lines = float(np.mean(lines)) if lines else 0.0
    mean_score = float(np.mean(scores)) if scores else 0.0
    mean_reward = float(np.mean(rewards)) if rewards else 0.0
    mean_steps = float(np.mean(steps_list)) if steps_list else 0.0
    top_out_rate = float(top_outs / max(len(lines), 1))
    mean_singles = float(np.mean(singles_list)) if singles_list else 0.0
    mean_doubles = float(np.mean(doubles_list)) if doubles_list else 0.0
    mean_triples = float(np.mean(triples_list)) if triples_list else 0.0
    mean_tetrises = float(np.mean(tetrises_list)) if tetrises_list else 0.0

    big_clear_bonus = (
        FITNESS_DOUBLE_BONUS * mean_doubles
        + FITNESS_TRIPLE_BONUS * mean_triples
        + FITNESS_TETRIS_BONUS * mean_tetrises
        - FITNESS_SINGLE_PENALTY * mean_singles
    )
    fitness = (
        mean_lines
        + FITNESS_SCORE_SCALE * mean_score
        + FITNESS_REWARD_SCALE * mean_reward
        + big_clear_bonus
        - 2.5 * top_out_rate
    )
    return EvalStats(
        fitness,
        mean_lines,
        mean_score,
        mean_reward,
        mean_steps,
        top_out_rate,
        mean_singles,
        mean_doubles,
        mean_triples,
        mean_tetrises,
    )


def _worker_eval(index: int, individual_data: dict[str, float], episodes: int, base_seed: int, max_steps: int, preset: str | None, lookahead_depth: int):
    individual = dict_to_individual(individual_data)
    stats = evaluate_individual(
        individual,
        episodes=episodes,
        base_seed=base_seed,
        max_steps=max_steps,
        preset=preset,
        lookahead_depth=lookahead_depth,
    )
    return index, individual_data, asdict(stats)


def evaluate_population(
    population: list[GAIndividual],
    *,
    episodes: int,
    base_seed: int,
    max_steps: int,
    preset: str | None,
    lookahead_depth: int,
    workers: int,
    logger: logging.Logger,
    executor: cf.ProcessPoolExecutor | None,
) -> tuple[list[float], list[EvalStats]]:
    fitnesses: list[float] = [0.0 for _ in population]
    stats_list: list[EvalStats] = [EvalStats(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0) for _ in population]

    if workers <= 1 or executor is None:
        for idx, individual in enumerate(population):
            stats = evaluate_individual(
                individual,
                episodes=episodes,
                base_seed=base_seed,
                max_steps=max_steps,
                preset=preset,
                lookahead_depth=lookahead_depth,
            )
            fitnesses[idx] = stats.fitness
            stats_list[idx] = stats
            logger.debug(
                f"  ind={idx:02d} | fitness={stats.fitness:8.3f} | lines={stats.mean_lines:6.2f} | score={stats.mean_score:8.2f} | S/D/T/T4={stats.mean_singles:.1f}/{stats.mean_doubles:.1f}/{stats.mean_triples:.1f}/{stats.mean_tetrises:.1f} | steps={stats.mean_steps:7.2f} | topout={stats.top_out_rate:.2%}"
            )
        return fitnesses, stats_list

    futures = []
    for idx, individual in enumerate(population):
        futures.append(
            executor.submit(
                _worker_eval,
                idx,
                individual_to_dict(individual),
                episodes,
                base_seed,
                max_steps,
                preset,
                lookahead_depth,
            )
        )

    for future in cf.as_completed(futures):
        idx, _individual_data, stats_dict = future.result()
        stats = EvalStats(**stats_dict)
        fitnesses[idx] = stats.fitness
        stats_list[idx] = stats
        logger.debug(
            f"  ind={idx:02d} | fitness={stats.fitness:8.3f} | lines={stats.mean_lines:6.2f} | score={stats.mean_score:8.2f} | S/D/T/T4={stats.mean_singles:.1f}/{stats.mean_doubles:.1f}/{stats.mean_triples:.1f}/{stats.mean_tetrises:.1f} | steps={stats.mean_steps:7.2f} | topout={stats.top_out_rate:.2%}"
        )
    return fitnesses, stats_list


def build_initial_population(args: argparse.Namespace, rng: random.Random) -> list[GAIndividual]:
    population = [GAIndividual()]
    if args.resume_from is not None:
        with open(args.resume_from, "r", encoding="utf-8") as f:
            data = json.load(f)
        payload = data["best_individual"] if "best_individual" in data else data
        population.append(dict_to_individual(payload))
    while len(population) < args.population:
        population.append(random_individual(rng))
    return population[: args.population]


def save_best_checkpoint(run_dir: Path, generation: int, best_individual: GAIndividual, best_stats: EvalStats) -> None:
    save_json(
        run_dir / "best_weights.json",
        {"generation": generation, "best_individual": individual_to_dict(best_individual), "stats": asdict(best_stats)},
    )
    save_json(
        run_dir / "best_model.json",
        {"generation": generation, "best_individual": individual_to_dict(best_individual), "stats": asdict(best_stats)},
    )
    save_json(
        run_dir / "checkpoints" / f"gen_{generation + 1:03d}_best.json",
        {"generation": generation, "best_individual": individual_to_dict(best_individual), "stats": asdict(best_stats)},
    )


def save_population_snapshot(run_dir: Path, generation: int, ranked) -> None:
    payload = [
        {"rank": i + 1, "individual": individual_to_dict(ind), "stats": asdict(stats), "fitness": fit}
        for i, (ind, stats, fit) in enumerate(ranked)
    ]
    save_json(run_dir / "population" / f"gen_{generation + 1:03d}.json", payload)


def text_generation_report(logger: logging.Logger, generation: int, ranked, topk: int) -> None:
    logger.info(f"Generation {generation + 1} leaderboard:")
    for idx, (ind, stats, fit) in enumerate(ranked[: max(1, topk)]):
        logger.info(
            f"  #{idx + 1}: fitness={fit:8.3f} | lines={stats.mean_lines:6.2f} | score={stats.mean_score:8.2f} | S/D/T/T4={stats.mean_singles:.1f}/{stats.mean_doubles:.1f}/{stats.mean_triples:.1f}/{stats.mean_tetrises:.1f} | steps={stats.mean_steps:7.2f} | topout={stats.top_out_rate:.1%} | lookahead={ind.lookahead_weight:.3f}"
        )


def generation_showcase(
    args: argparse.Namespace,
    run_dir: Path,
    generation: int,
    baseline: GAIndividual,
    current_best: GAIndividual,
    best_overall: GAIndividual | None,
    window,
) -> bool:
    _, ShowcaseVideoWriter, _, compose_panels, draw_board_panel = _viz_modules()
    if best_overall is None:
        best_overall = current_best
    triples = [
        ("Seed baseline", baseline),
        (f"Gen {generation + 1} best", current_best),
        ("Best overall", best_overall),
    ]
    envs = [
        PlacementTetrisEnv(reward_config=ind.to_reward_config(), max_steps=args.max_steps, initial_board_preset=args.preset)
        for _, ind in triples
    ]
    video_writer = None
    if args.showcase_save_video:
        video_path = run_dir / "showcase_videos" / f"gen_{generation + 1:03d}.mp4"
        video_writer = ShowcaseVideoWriter(video_path, fps=args.showcase_video_fps)

    seed = args.seed + generation * 1000 + 7
    infos = []
    terms = []
    truncs = []
    totals = []
    weights = []
    try:
        for idx, (_, ind) in enumerate(triples):
            _, info = envs[idx].reset(seed=seed)
            infos.append(info)
            terms.append(False)
            truncs.append(False)
            totals.append({"score": 0.0, "lines": 0, "steps": 0})
            weights.append(ind.to_heuristic_weights())

        for step in range(args.showcase_steps):
            panels = []
            active_any = False
            for idx, (title, ind) in enumerate(triples):
                env = envs[idx]
                if not (terms[idx] or truncs[idx]):
                    action, _ = choose_action(env, weights[idx], args.lookahead_depth, ind.lookahead_weight)
                    _, _, terms[idx], truncs[idx], infos[idx] = env.step(action)
                    totals[idx]["score"] = float(infos[idx].get("total_score", 0.0))
                    totals[idx]["lines"] = int(infos[idx].get("total_lines_cleared", 0))
                    totals[idx]["steps"] += 1
                    active_any = True
                status = "done" if (terms[idx] or truncs[idx]) else f"step {totals[idx]['steps']}"
                panels.append(
                    draw_board_panel(
                        board=env.board,
                        current_piece=env.current_piece,
                        next_piece=env.next_piece,
                        title=title,
                        lines=totals[idx]["lines"],
                        score=totals[idx]["score"],
                        steps=totals[idx]["steps"],
                        status=status,
                        cell_px=args.showcase_upscale,
                    )
                )
            frame = compose_panels(panels)
            if args.showcase_save_frames and step == args.showcase_steps - 1:
                save_path = run_dir / "showcase_frames" / f"gen_{generation + 1:03d}.png"
                save_path.parent.mkdir(parents=True, exist_ok=True)
                import cv2
                cv2.imwrite(str(save_path), frame)
            if video_writer is not None:
                video_writer.write(frame)
            quit_requested = window.show(frame, delay_ms=args.showcase_delay_ms) if args.live_boards else False
            if quit_requested:
                return True
            if not active_any:
                break
    finally:
        for env in envs:
            env.close()
        if video_writer is not None:
            video_writer.close()
    return False


def main() -> None:
    args = parse_args()
    if args.population < 2:
        raise ValueError("population must be at least 2")
    if args.elite < 1 or args.elite >= args.population:
        raise ValueError("elite must be in [1, population - 1]")
    if args.workers < 1:
        raise ValueError("workers must be at least 1")

    run_dir = RUNS_DIR / args.save_name
    run_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(run_dir, verbosity=args.verbosity)
    logger.info(f"GA run: {args.save_name}")
    logger.info(f"Population: {args.population}, Generations: {args.generations}, Workers: {args.workers}")
    save_json(run_dir / "ga_config.json", vars(args))

    rng = random.Random(args.seed)
    population = build_initial_population(args, rng)

    history: list[dict[str, float]] = []
    best_overall: tuple[GAIndividual, EvalStats] | None = None
    baseline = GAIndividual()
    LiveDashboard, _, ShowcaseWindow, _, _ = _viz_modules() if (args.live_dashboard or args.live_boards or args.showcase_save_frames or args.showcase_save_video) else (None, None, None, None, None)
    dashboard = LiveDashboard(run_dir) if args.live_dashboard and LiveDashboard is not None else None
    showcase_window = ShowcaseWindow(enabled=args.live_boards) if ShowcaseWindow is not None else type("_NullShowcase", (), {"show": lambda self, frame, delay_ms=0: False, "close": lambda self: None})()

    executor = None
    if args.workers > 1:
        ctx = mp.get_context("spawn")
        executor = cf.ProcessPoolExecutor(max_workers=args.workers, mp_context=ctx)

    try:
        for generation in range(args.generations):
            logger.info(f"\n{'='*68}")
            logger.info(f"Generation {generation + 1}/{args.generations}")
            logger.info(f"{'='*68}")

            generation_seed = args.seed + generation * 10_000
            fitnesses, stats_list = evaluate_population(
                population,
                episodes=args.episodes_per_individual,
                base_seed=generation_seed,
                max_steps=args.max_steps,
                preset=args.preset,
                lookahead_depth=args.lookahead_depth,
                workers=args.workers,
                logger=logger,
                executor=executor,
            )

            ranked = sorted(zip(population, stats_list, fitnesses), key=lambda item: item[2], reverse=True)
            best_individual, best_stats, _ = ranked[0]

            text_generation_report(logger, generation, ranked, args.text_topk)

            if best_overall is None or best_stats.fitness > best_overall[1].fitness:
                best_overall = (best_individual, best_stats)
                save_best_checkpoint(run_dir, generation, best_individual, best_stats)
                logger.info(f"  NEW BEST: fitness={best_stats.fitness:.3f} (gen {generation + 1})")

            gen_summary = {
                "generation": generation,
                "best_fitness": float(best_stats.fitness),
                "mean_fitness": float(np.mean(fitnesses)),
                "std_fitness": float(np.std(fitnesses)),
                "best_mean_lines": float(best_stats.mean_lines),
                "best_mean_score": float(best_stats.mean_score),
                "best_top_out_rate": float(best_stats.top_out_rate),
                "best_mean_singles": float(best_stats.mean_singles),
                "best_mean_doubles": float(best_stats.mean_doubles),
                "best_mean_triples": float(best_stats.mean_triples),
                "best_mean_tetrises": float(best_stats.mean_tetrises),
            }
            history.append(gen_summary)
            save_json(run_dir / "history.json", history)

            logger.info(
                f"Gen summary: best={best_stats.fitness:.3f} | mean={np.mean(fitnesses):.3f} ± {np.std(fitnesses):.3f} | lines={best_stats.mean_lines:.2f} | score={best_stats.mean_score:.2f} | S/D/T/T4={best_stats.mean_singles:.1f}/{best_stats.mean_doubles:.1f}/{best_stats.mean_triples:.1f}/{best_stats.mean_tetrises:.1f} | topout={best_stats.top_out_rate:.1%}"
            )

            if args.save_population_snapshots:
                save_population_snapshot(run_dir, generation, ranked)

            if dashboard is not None:
                dashboard.update(history)
                dashboard.save_png(f"dashboard_gen_{generation + 1:03d}.png")

            if args.plot and args.plot_every > 0 and (generation + 1) % args.plot_every == 0:
                plot_fitness_evolution(history, run_dir, logger)
                plot_metrics_evolution(history, run_dir, logger)
                plot_clear_mix_evolution(history, run_dir, logger)

            if (args.live_boards or args.showcase_save_frames or args.showcase_save_video) and (generation % max(1, args.showcase_every) == 0):
                quit_requested = generation_showcase(
                    args=args,
                    run_dir=run_dir,
                    generation=generation,
                    baseline=baseline,
                    current_best=best_individual,
                    best_overall=best_overall[0] if best_overall is not None else None,
                    window=showcase_window,
                )
                if quit_requested:
                    logger.info("Showcase window requested quit; disabling live boards for the rest of the run.")
                    args.live_boards = False

            elites = [individual for individual, _, _ in ranked[: args.elite]]
            next_population = elites.copy()

            while len(next_population) < args.population:
                parent_a = tournament_select(population, fitnesses, rng, args.tournament_size)
                parent_b = tournament_select(population, fitnesses, rng, args.tournament_size)
                child = blend_crossover(parent_a, parent_b, rng, args.blend_alpha)
                child = mutate(child, rng, args.mutation_rate, args.mutation_scale)
                next_population.append(child)
            population = next_population

    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=False)
        if dashboard is not None:
            dashboard.save_png()
            dashboard.close()
        showcase_window.close()
        try:
            import cv2
            cv2.destroyAllWindows()
        except Exception:
            pass

    if best_overall is None:
        raise RuntimeError("GA finished without evaluating any individuals.")

    best_individual, best_stats = best_overall
    best_dict = individual_to_dict(best_individual)
    logger.info(f"\n{'='*68}")
    logger.info("GA Run Complete")
    logger.info(f"{'='*68}")
    for key, value in best_dict.items():
        logger.info(f"  {key}: {value:.4f}")
    for key, value in asdict(best_stats).items():
        logger.info(f"  {key}: {value}")

    if not args.no_history_csv:
        csv_path = run_dir / "history.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            if history:
                writer = csv.DictWriter(f, fieldnames=history[0].keys())
                writer.writeheader()
                writer.writerows(history)
        logger.info(f"Saved history to {csv_path.name}")

    if args.plot:
        plot_fitness_evolution(history, run_dir, logger)
        plot_metrics_evolution(history, run_dir, logger)
        plot_clear_mix_evolution(history, run_dir, logger)
        plot_final_weights(best_dict, run_dir, logger)

    logger.info(f"Saved all outputs to: {run_dir}")


if __name__ == "__main__":
    main()
