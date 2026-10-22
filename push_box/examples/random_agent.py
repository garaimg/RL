"""Watch a random agent using the ANSI or PyGame renderer."""

import argparse

from push_box.pushbox_env import PushBoxEnv


def main() -> None:
    parser = argparse.ArgumentParser()
    renderer_group = parser.add_mutually_exclusive_group()
    renderer_group.add_argument("--human", action="store_true", help="use PyGame")
    renderer_group.add_argument("--mujoco", action="store_true", help="use MuJoCo")
    args = parser.parse_args()

    render_mode = "human" if args.human or args.mujoco else "ansi"
    renderer = "mujoco" if args.mujoco else "pygame"
    env = PushBoxEnv(render_mode=render_mode, renderer=renderer, max_steps=50)
    env.reset(seed=0)
    render(env)

    terminated = False
    truncated = False
    while not (terminated or truncated):
        action = env.action_space.sample()
        _, _, terminated, truncated, _ = env.step(action)
        render(env)

    env.close()

def render(env: PushBoxEnv) -> None:
    """Render the environment and print the result only in ANSI mode."""
    rendering = env.render()
    if rendering is not None:
        print(rendering)


if __name__ == "__main__":
    main()
