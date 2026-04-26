from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.utils import get_action_masks

from src.envs.placement_env import PlacementTetrisEnv, RewardConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Watch a saved MaskablePPO model play placement Tetris.")
    parser.add_argument("--model-path", type=str, required=True, help="Path to saved .zip model")
    parser.add_argument("--episodes", type=int, default=3, help="Number of episodes to watch")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed")
    parser.add_argument("--device", type=str, default="auto", help="Device: auto, cuda, cpu")
    parser.add_argument("--delay-ms", type=int, default=120, help="Delay between frames")
    parser.add_argument("--upscale", type=int, default=30, help="Render upscale")
    parser.add_argument("--max-steps", type=int, default=500, help="Episode truncation limit")
    parser.add_argument("--preset", type=str, default=None, help="Optional preset board name")
    return parser.parse_args()


def make_reward_config() -> RewardConfig:
    return RewardConfig(
        single_score=2.0,
        double_score=8.0,
        triple_score=32.0,
        tetris_score=120.0,
        shaping_scale=0.02,
        shaping_gamma=0.99,
        aggregate_height_weight=0.12,
        holes_weight=3.00,
        bumpiness_weight=0.08,
        max_height_weight=0.18,
        row_transitions_weight=0.50,
        column_transitions_weight=0.35,
        rows_with_holes_weight=1.00,
        hole_depth_weight=0.50,
        cumulative_wells_weight=0.60,
        eroded_piece_cells_weight=1.25,
        survival_bonus=0.0,
        game_over_penalty=-25.0,
        invalid_action_penalty=-5.0,
    )


def make_env(max_steps: int, upscale: int, preset: str | None):
    kwargs = {
        "reward_config": make_reward_config(),
        "max_steps": max_steps,
        "render_mode": "rgb_array",
        "render_upscale": upscale,
    }
    if preset is not None:
        kwargs["initial_board_preset"] = preset
    return PlacementTetrisEnv(**kwargs)


def main() -> None:
    args = parse_args()
    model_path = Path(args.model_path)

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    print(f"Loading model from: {model_path}")
    model = MaskablePPO.load(str(model_path), device=args.device)

    env = make_env(
        max_steps=args.max_steps,
        upscale=args.upscale,
        preset=args.preset,
    )

    try:
        for ep in range(args.episodes):
            obs, info = env.reset(seed=args.seed + ep)
            terminated = False
            truncated = False
            total_reward = 0.0
            steps = 0

            print(
                f"\nStarting episode {ep + 1} | "
                f"current={info['current_piece']} next={info['next_piece']} "
                f"valid_actions={info['valid_action_count']}"
            )

            frame = env.render()
            if frame is not None:
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                cv2.imshow("Saved MaskablePPO Model", frame_bgr)
                if (cv2.waitKey(args.delay_ms) & 0xFF) == ord("q"):
                    print("Quit requested.")
                    return

            while not (terminated or truncated):
                action_masks = get_action_masks(env)
                action, _ = model.predict(obs, action_masks=action_masks, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(int(action))

                total_reward += float(reward)
                steps += 1

                frame = env.render()
                if frame is not None:
                    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    cv2.imshow("Saved MaskablePPO Model", frame_bgr)
                    if (cv2.waitKey(args.delay_ms) & 0xFF) == ord("q"):
                        print("Quit requested.")
                        return

            print(
                f"Episode {ep + 1} finished | "
                f"reward={total_reward:.2f} | "
                f"steps={steps} | "
                f"score={info.get('total_score', 0.0):.1f} | "
                f"lines_total={info.get('total_lines_cleared', 0)} | "
                f"terminated={terminated} truncated={truncated}"
            )

    finally:
        env.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()