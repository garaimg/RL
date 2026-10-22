"""Train PushBox with the course's tabular Q-learning implementation."""

import argparse
import random

import numpy as np

from deustorl.common import EpsilonGreedyPolicy, evaluate_policy, max_policy
from deustorl.qlearning import QLearning
from push_box.pushbox_env import PushBoxEnv


def main() -> None:
    parser = argparse.ArgumentParser()
    renderer_group = parser.add_mutually_exclusive_group()
    renderer_group.add_argument("--human", action="store_true", help="use PyGame")
    renderer_group.add_argument("--mujoco", action="store_true", help="use MuJoCo")
    args = parser.parse_args()

    seed = 3
    random.seed(seed)
    np.random.seed(seed)

    training_env = PushBoxEnv(max_steps=200)
    algorithm = QLearning(training_env)
    exploration_policy = EpsilonGreedyPolicy(epsilon=0.1)

    training_env.reset(seed=seed)

    print("Training Q-learning algorithm...")
    algorithm.learn(exploration_policy, n_steps=200_000, discount_rate=0.99, lr=0.1)

    training_env.close()

    print("Evaluating learned policy...")
    render_mode = "human" if args.human or args.mujoco else "ansi"
    renderer = "mujoco" if args.mujoco else "pygame"
    evaluation_env = PushBoxEnv(render_mode=render_mode, renderer=renderer, max_steps=200)
    evaluate_policy(evaluation_env, algorithm.q_table, max_policy, n_episodes=3, verbose=True)

    evaluation_env.close()


if __name__ == "__main__":
    main()
