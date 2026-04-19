import numpy as np

from src.envs.placement_env import PlacementTetrisEnv


def main() -> None:
    env = PlacementTetrisEnv()

    obs, info = env.reset(seed=42)
    print("Reset info:", info)
    print("Board shape:", obs["board"].shape)
    print("Current piece index:", obs["current_piece"])
    print("Next piece index:", obs["next_piece"])
    print("Valid action count:", int(np.sum(obs["action_mask"])))

    for step in range(5):
        valid_actions = np.flatnonzero(env.action_masks())
        if len(valid_actions) == 0:
            print("No valid actions left.")
            break

        action = int(valid_actions[0])
        obs, reward, terminated, truncated, info = env.step(action)

        print(
            f"step={step} action={action} reward={reward:.3f} "
            f"valid={info['valid_action_count']} "
            f"lines_total={info['total_lines_cleared']} "
            f"terminated={terminated} truncated={truncated}"
        )

        ansi = env.render()
        if ansi is not None:
            print(ansi)
            print("-" * 40)

        if terminated or truncated:
            break

    env.close()


if __name__ == "__main__":
    main()