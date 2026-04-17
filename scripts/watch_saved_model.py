from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import gymnasium as gym
from stable_baselines3 import DQN
from tetris_gymnasium.envs.tetris import Tetris


ENV_ID = "tetris_gymnasium/Tetris"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Watch a saved DQN model play Tetris.")
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to the saved SB3 model .zip file",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=3,
        help="Number of episodes to watch",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base random seed",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to load model on: auto, cuda, cpu",
    )
    parser.add_argument(
        "--delay-ms",
        type=int,
        default=120,
        help="Delay between frames in milliseconds",
    )
    parser.add_argument(
        "--upscale",
        type=int,
        default=30,
        help="Render upscale factor",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Use deterministic actions",
    )
    return parser.parse_args()


def make_env(upscale: int):
    return gym.make(
        ENV_ID,
        render_mode="rgb_array",
        render_upscale=upscale,
    )


def main() -> None:
    args = parse_args()
    model_path = Path(args.model_path)

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    print(f"Loading model from: {model_path}")
    model = DQN.load(str(model_path), device=args.device)

    env = make_env(upscale=args.upscale)

    try:
        for ep in range(args.episodes):
            obs, info = env.reset(seed=args.seed + ep)
            done = False
            total_reward = 0.0
            steps = 0

            print(f"\nStarting episode {ep + 1}")

            while not done:
                frame = env.render()
                if frame is not None:
                    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    cv2.imshow("Saved Tetris Model", frame_bgr)

                action, _ = model.predict(obs, deterministic=args.deterministic)
                obs, reward, terminated, truncated, info = env.step(action)

                total_reward += float(reward)
                steps += 1
                done = terminated or truncated

                key = cv2.waitKey(args.delay_ms) & 0xFF
                if key == ord("q"):
                    print("Quit requested.")
                    return

            print(
                f"Episode {ep + 1} finished | "
                f"reward={total_reward:.2f} | "
                f"steps={steps} | "
                f"lines_cleared={info.get('lines_cleared', 0)}"
            )

    finally:
        env.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()