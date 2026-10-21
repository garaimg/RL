import os
import gymnasium as gym
import random

from deustorl.common import *
from deustorl.sarsa import Sarsa
from deustorl.qlearning import QLearning
from deustorl.expected_sarsa import ExpectedSarsa

from deustorl.helpers import DiscretizedObservationWrapper

def train_and_evaluate(algo, n_steps= 60000, **kwargs):
    epsilon_greedy_policy = EpsilonGreedyPolicy(epsilon=0.1)
    start_time = time.time()
    algo.learn(epsilon_greedy_policy, n_steps, **kwargs)
    print("----- {:0.4f} secs. -----".format(time.time() - start_time))

    return evaluate_policy(algo.env, algo.q_table, max_policy, n_episodes=100, verbose=False)


os.system("rm -rf ./logs/")

#env_name = "CartPole-v1"
env_name = "MountainCar-v0" # MountainCar-v0 best score is -110.0

env = DiscretizedObservationWrapper(gym.make(env_name), n_bins=10)
visual_env = DiscretizedObservationWrapper(gym.make(env_name, render_mode='human'), n_bins=10)

seed = 3
random.seed(seed)
env.reset(seed=seed)

n_steps = 500_000

algo = Sarsa(env)
print("Testing SARSA")
train_and_evaluate(algo, n_steps=n_steps, lr=0.1)
evaluate_policy(visual_env, algo.q_table, max_policy, n_episodes=10, verbose=False)

algo = QLearning(env)
print("Testing Q-Learning")
train_and_evaluate(algo, n_steps=n_steps, lr=0.1)
evaluate_policy(visual_env, algo.q_table, max_policy, n_episodes=10, verbose=False)

algo = ExpectedSarsa(env)
print("Testing Expected SARSA")
train_and_evaluate(algo, n_steps=n_steps, lr=0.1)
evaluate_policy(visual_env, algo.q_table, max_policy, n_episodes=10, verbose=False)


