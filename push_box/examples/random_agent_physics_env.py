"""Watch physical transitions between the discrete PushBox states."""

from push_box.pushbox_physics_env import PushBoxPhysicsEnv


def main() -> None:
    env = PushBoxPhysicsEnv(render_mode="human", max_steps=50)
    env.reset(seed=0)

    terminated = False
    truncated = False
    while not (terminated or truncated):
        action = env.action_space.sample()
        _, _, terminated, truncated, _ = env.step(action)

    env.close()


if __name__ == "__main__":
    main()
