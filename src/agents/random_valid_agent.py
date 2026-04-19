from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import matplotlib.pyplot as plt
import numpy as np

from src.envs.placement_env import PlacementTetrisEnv


PLOTS_DIR = PROJECT_ROOT / "runs" / "plots"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a random valid-action agent on PlacementTetrisEnv.")
    parser.add_argument("--episodes", type=int, default=25, help="Number of episodes to run.")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed.")
    parser.add_argument("--max-steps", type=int, default=500, help="Episode truncation limit inside the env.")
    parser.add_argument("--preset", type=str, default=None, help="Optional preset board name.")
    parser.add_argument("--render", action="store_true", help="Show visual Tetris window (OpenCV).")
    parser.add_argument("--delay-ms", type=int, default=120, help="Delay between frames when rendering.")
    parser.add_argument("--upscale", type=int, default=30, help="Render upscale factor.")
    parser.add_argument("--print-board", action="store_true", help="Print ANSI board after each step.")
    parser.add_argument("--plot", action="store_true", help="Save summary plots.")
    parser.add_argument("--save-prefix", type=str, default="random_valid_agent", help="Prefix for saved plot filenames.")
    return parser.parse_args()


def maybe_show_frame(frame: np.ndarray | None, delay_ms: int) -> bool:
    if frame is None:
        return False

    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    cv2.imshow("Placement Tetris", frame_bgr)

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


def maybe_save_plots(
    rewards: np.ndarray,
    lengths: np.ndarray,
    total_lines: np.ndarray,
    prefix: str,
) -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    plot_series(
        values=rewards,
        ylabel="Episode Reward",
        title="Random Valid Agent: Episode Reward",
        save_path=PLOTS_DIR / f"{prefix}_rewards.png",
    )
    plot_series(
        values=lengths,
        ylabel="Episode Length",
        title="Random Valid Agent: Episode Length",
        save_path=PLOTS_DIR / f"{prefix}_lengths.png",
    )
    plot_series(
        values=total_lines,
        ylabel="Total Lines Cleared",
        title="Random Valid Agent: Total Lines Cleared",
        save_path=PLOTS_DIR / f"{prefix}_lines.png",
    )

    print(f"\nSaved plots to: {PLOTS_DIR.resolve()}")


def main() -> None:
    args = parse_args()

    render_mode = None
    if args.render:
        render_mode = "rgb_array"
    elif args.print_board:
        render_mode = "ansi"

    env_kwargs = {
        "max_steps": args.max_steps,
        "render_mode": render_mode,
        "render_upscale": args.upscale,
    }
    if args.preset is not None:
        env_kwargs["initial_board_preset"] = args.preset

    env = PlacementTetrisEnv(**env_kwargs)

    episode_rewards: list[float] = []
    episode_lengths: list[int] = []
    episode_total_lines: list[int] = []
    episode_invalid_flags: list[int] = []

    try:
        for ep in range(args.episodes):
            obs, info = env.reset(seed=args.seed + ep)
            terminated = False
            truncated = False
            total_reward = 0.0
            steps = 0

            print(
                f"\nEpisode {ep + 1:02d} start | "
                f"current={info['current_piece']} next={info['next_piece']} "
                f"valid_actions={info['valid_action_count']}"
            )

            # Show initial state once
            if args.render:
                frame = env.render()
                if maybe_show_frame(frame, args.delay_ms):
                    raise KeyboardInterrupt("Quit requested by user.")

            while not (terminated or truncated):
                valid_actions = np.flatnonzero(env.action_masks())

                if valid_actions.size == 0:
                    print("No valid actions available.")
                    break

                action = int(env.np_random.choice(valid_actions))
                obs, reward, terminated, truncated, info = env.step(action)

                total_reward += float(reward)
                steps += 1

                # For a placement env, render AFTER the step so you see the new board.
                if args.render:
                    frame = env.render()
                    if maybe_show_frame(frame, args.delay_ms):
                        raise KeyboardInterrupt("Quit requested by user.")

                if args.print_board:
                    ansi = env.render()
                    if ansi is not None:
                        print(ansi)
                        print("-" * 40)

            episode_rewards.append(total_reward)
            episode_lengths.append(steps)
            episode_total_lines.append(int(info["total_lines_cleared"]))
            episode_invalid_flags.append(int(bool(info.get("invalid_action", False))))

            print(
                f"Episode {ep + 1:02d} end   | "
                f"reward={total_reward:8.3f} | "
                f"steps={steps:4d} | "
                f"lines_total={info['total_lines_cleared']:3d} | "
                f"terminated={terminated} truncated={truncated}"
            )

    except KeyboardInterrupt as exc:
        print(f"\nStopped early: {exc}")

    finally:
        env.close()
        cv2.destroyAllWindows()

    if not episode_rewards:
        print("\nNo completed episodes to summarize.")
        return

    rewards = np.array(episode_rewards, dtype=np.float32)
    lengths = np.array(episode_lengths, dtype=np.int32)
    total_lines = np.array(episode_total_lines, dtype=np.int32)
    invalids = np.array(episode_invalid_flags, dtype=np.int32)

    print("\nSummary")
    print(f"Episodes:            {len(rewards)}")
    print(f"Mean reward:         {rewards.mean():.3f}")
    print(f"Std reward:          {rewards.std():.3f}")
    print(f"Max reward:          {rewards.max():.3f}")
    print(f"Mean episode length: {lengths.mean():.2f}")
    print(f"Max episode length:  {lengths.max()}")
    print(f"Mean total lines:    {total_lines.mean():.2f}")
    print(f"Max total lines:     {total_lines.max()}")
    print(f"Invalid episodes:    {invalids.sum()}")

    if args.plot:
        maybe_save_plots(
            rewards=rewards,
            lengths=lengths,
            total_lines=total_lines,
            prefix=args.save_prefix,
        )


if __name__ == "__main__":
    main()