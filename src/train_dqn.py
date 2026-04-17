from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from tetris_gymnasium.envs.tetris import Tetris  # ensures env registration


ENV_ID = "tetris_gymnasium/Tetris"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a first-pass DQN on Tetris.")
    parser.add_argument("--timesteps", type=int, default=100_000, help="Total training timesteps.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--device", type=str, default="auto", help="Device: auto, cuda, cpu.")
    parser.add_argument("--eval-freq", type=int, default=10_000, help="Evaluation frequency in steps.")
    parser.add_argument("--save-name", type=str, default="dqn_tetris_baseline", help="Run/model name.")
    parser.add_argument("--watch", action="store_true", help="Watch a few episodes after training.")
    parser.add_argument("--watch-episodes", type=int, default=3, help="How many watch episodes to run.")
    parser.add_argument("--delay-ms", type=int, default=120, help="Frame delay when watching.")
    parser.add_argument("--upscale", type=int, default=30, help="Render upscale when watching.")
    return parser.parse_args()


def make_env(render: bool = False, upscale: int = 30):
    if render:
        env = gym.make(
            ENV_ID,
            render_mode="rgb_array",
            render_upscale=upscale,
        )
    else:
        env = gym.make(ENV_ID)

    env = Monitor(env)
    return env


def build_model(env, device: str, tb_log_dir: Path) -> DQN:
    model = DQN(
        policy="MultiInputPolicy",
        env=env,
        learning_rate=1e-4,
        buffer_size=100_000,
        learning_starts=5_000,
        batch_size=256,
        gamma=0.99,
        train_freq=4,
        gradient_steps=1,
        target_update_interval=1_000,
        exploration_fraction=0.25,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.05,
        policy_kwargs=dict(
            net_arch=[256, 256],
        ),
        tensorboard_log=str(tb_log_dir),
        verbose=1,
        device=device,
        seed=42,
    )
    return model


def watch_agent(model_path: Path, episodes: int, seed: int, delay_ms: int, upscale: int, device: str) -> None:
    env = make_env(render=True, upscale=upscale)
    model = DQN.load(str(model_path), device=device)

    try:
        for ep in range(episodes):
            obs, info = env.reset(seed=seed + ep)
            done = False
            total_reward = 0.0
            steps = 0

            print(f"Watching episode {ep + 1}")

            while not done:
                frame = env.render()
                if frame is not None:
                    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    cv2.imshow("Tetris DQN", frame_bgr)

                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)

                total_reward += float(reward)
                steps += 1
                done = terminated or truncated

                key = cv2.waitKey(delay_ms) & 0xFF
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


def main() -> None:
    args = parse_args()

    run_dir = Path("runs") / args.save_name
    tb_dir = run_dir / "tb"
    model_dir = run_dir / "models"
    checkpoint_dir = run_dir / "checkpoints"
    best_model_dir = run_dir / "best_model"

    run_dir.mkdir(parents=True, exist_ok=True)
    tb_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_model_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "env_id": ENV_ID,
        "timesteps": args.timesteps,
        "seed": args.seed,
        "device": args.device,
        "eval_freq": args.eval_freq,
        "save_name": args.save_name,
        "policy": "MultiInputPolicy",
        "algorithm": "DQN",
        "learning_rate": 1e-4,
        "buffer_size": 100_000,
        "learning_starts": 5_000,
        "batch_size": 256,
        "gamma": 0.99,
        "train_freq": 4,
        "gradient_steps": 1,
        "target_update_interval": 1_000,
        "exploration_fraction": 0.25,
        "exploration_initial_eps": 1.0,
        "exploration_final_eps": 0.05,
        "net_arch": [256, 256],
    }
    with open(run_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    train_env = make_env(render=False)
    eval_env = make_env(render=False)

    print("Train action space:", train_env.action_space)
    print("Train observation space:", train_env.observation_space)

    model = build_model(train_env, device=args.device, tb_log_dir=tb_dir)

    checkpoint_callback = CheckpointCallback(
        save_freq=25_000,
        save_path=str(checkpoint_dir),
        name_prefix="dqn_checkpoint",
        save_replay_buffer=True,
        save_vecnormalize=False,
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(best_model_dir),
        log_path=str(run_dir / "eval_logs"),
        eval_freq=args.eval_freq,
        n_eval_episodes=5,
        deterministic=True,
        render=False,
    )

    model.learn(
        total_timesteps=args.timesteps,
        callback=[checkpoint_callback, eval_callback],
        log_interval=10,
        progress_bar=True,
    )

    final_model_path = model_dir / "dqn_tetris_final"
    model.save(str(final_model_path))
    print(f"\nSaved final model to: {final_model_path}")

    train_env.close()
    eval_env.close()

    best_model_path = best_model_dir / "best_model.zip"
    if args.watch:
        if best_model_path.exists():
            print("\nWatching best saved model...")
            watch_agent(
                model_path=best_model_path,
                episodes=args.watch_episodes,
                seed=args.seed,
                delay_ms=args.delay_ms,
                upscale=args.upscale,
                device=args.device,
            )
        else:
            print("\nBest model not found yet, watching final model instead...")
            watch_agent(
                model_path=final_model_path.with_suffix(".zip"),
                episodes=args.watch_episodes,
                seed=args.seed,
                delay_ms=args.delay_ms,
                upscale=args.upscale,
                device=args.device,
            )


if __name__ == "__main__":
    main()