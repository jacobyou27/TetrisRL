from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from tetris_gymnasium.envs.tetris import Tetris  # ensures env registration


ENV_ID = "tetris_gymnasium/Tetris"
PLOTS_DIR = Path("runs/plots")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a random agent in Tetris.")
    parser.add_argument("--episodes", type=int, default=25, help="Number of episodes to run.")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed.")
    parser.add_argument("--render", action="store_true", help="Show a visual Tetris window.")
    parser.add_argument("--plot", action="store_true", help="Save plots after running.")
    parser.add_argument("--delay-ms", type=int, default=120, help="Delay between frames when rendering.")
    parser.add_argument("--upscale", type=int, default=30, help="Render upscale factor.")
    return parser.parse_args()


def make_env(render: bool, upscale: int):
    if render:
        return gym.make(
            ENV_ID,
            render_mode="rgb_array",
            render_upscale=upscale,
        )
    return gym.make(ENV_ID)


def maybe_show_frame(env, delay_ms: int) -> bool:
    frame = env.render()
    if frame is not None:
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.imshow("Tetris", frame_bgr)

    key = cv2.waitKey(delay_ms) & 0xFF
    return key == ord("q")


def run_episode(env, seed: int, render: bool, delay_ms: int) -> dict[str, float | int]:
    obs, info = env.reset(seed=seed)
    done = False
    total_reward = 0.0
    steps = 0
    last_lines_cleared = 0

    while not done:
        if render:
            should_quit = maybe_show_frame(env, delay_ms)
            if should_quit:
                raise KeyboardInterrupt("Quit requested by user.")

        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        total_reward += float(reward)
        steps += 1
        last_lines_cleared = int(info.get("lines_cleared", last_lines_cleared))
        done = terminated or truncated

    return {
        "reward": total_reward,
        "steps": steps,
        "lines_cleared": last_lines_cleared,
    }


def plot_series(values: np.ndarray, ylabel: str, title: str, save_path: Path) -> None:
    plt.figure(figsize=(8, 4.5))
    plt.plot(np.arange(1, len(values) + 1), values)
    plt.xlabel("Episode")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def save_plots(rewards: np.ndarray, lengths: np.ndarray, lines: np.ndarray) -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    plot_series(
        values=rewards,
        ylabel="Total Reward",
        title="Random Agent: Episode Reward",
        save_path=PLOTS_DIR / "random_agent_rewards.png",
    )
    plot_series(
        values=lengths,
        ylabel="Episode Length",
        title="Random Agent: Episode Length",
        save_path=PLOTS_DIR / "random_agent_lengths.png",
    )
    plot_series(
        values=lines,
        ylabel="Lines Cleared",
        title="Random Agent: Lines Cleared",
        save_path=PLOTS_DIR / "random_agent_lines.png",
    )

    print(f"\nSaved plots to: {PLOTS_DIR.resolve()}")


def main() -> None:
    args = parse_args()
    env = make_env(render=args.render, upscale=args.upscale)

    print("Action space:", env.action_space)
    print("Observation space:", env.observation_space)

    episode_rewards: list[float] = []
    episode_lengths: list[int] = []
    episode_lines: list[int] = []

    try:
        for ep in range(args.episodes):
            result = run_episode(
                env=env,
                seed=args.seed + ep,
                render=args.render,
                delay_ms=args.delay_ms,
            )

            episode_rewards.append(float(result["reward"]))
            episode_lengths.append(int(result["steps"]))
            episode_lines.append(int(result["lines_cleared"]))

            print(
                f"Episode {ep + 1:02d} | "
                f"reward={result['reward']:7.2f} | "
                f"steps={result['steps']:4d} | "
                f"lines_cleared={result['lines_cleared']}"
            )

    except KeyboardInterrupt as exc:
        print(f"\nStopped early: {exc}")

    finally:
        env.close()
        cv2.destroyAllWindows()

    if not episode_rewards:
        print("No completed episodes to summarize.")
        return

    rewards = np.array(episode_rewards, dtype=np.float32)
    lengths = np.array(episode_lengths, dtype=np.int32)
    lines = np.array(episode_lines, dtype=np.int32)

    print("\nSummary")
    print(f"Mean reward:        {rewards.mean():.2f}")
    print(f"Std reward:         {rewards.std():.2f}")
    print(f"Max reward:         {rewards.max():.2f}")
    print(f"Mean episode len:   {lengths.mean():.2f}")
    print(f"Max episode len:    {lengths.max()}")
    print(f"Mean lines cleared: {lines.mean():.2f}")
    print(f"Max lines cleared:  {lines.max()}")

    if args.plot:
        save_plots(rewards, lengths, lines)


if __name__ == "__main__":
    main()