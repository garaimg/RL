"""Train Q-learning and evaluate it with physical transitions."""

import random

import numpy as np

from deustorl.common import EpsilonGreedyPolicy, evaluate_policy, max_policy
from deustorl.qlearning import QLearning
from push_box.pushbox_env import PushBoxEnv
from push_box.pushbox_physics_env import PushBoxPhysicsEnv


def main() -> None:
    seed = 3
    random.seed(seed)
    np.random.seed(seed)

    # The discrete MDP is identical, so invisible physics is unnecessary while training.
    training_env = PushBoxEnv(max_steps=200)
    algorithm = QLearning(training_env)
    exploration_policy = EpsilonGreedyPolicy(epsilon=0.1)

    training_env.reset(seed=seed)
    print("Training Q-learning algorithm...")
    algorithm.learn(exploration_policy, n_steps=200_000, discount_rate=0.99, lr=0.1)
    training_env.close()

    print("Evaluating learned policy with physical transitions...")
    evaluation_env = PushBoxPhysicsEnv(render_mode="human", max_steps=200)
    evaluation_env.reset(seed=seed)
    evaluate_policy(evaluation_env, algorithm.q_table, max_policy, n_episodes=3)
    evaluation_env.close()


if __name__ == "__main__":
    main()
