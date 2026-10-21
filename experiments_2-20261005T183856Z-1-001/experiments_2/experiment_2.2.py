import os
import gymnasium as gym
from gymnasium.envs.toy_text.frozen_lake import generate_random_map
import numpy as np

from deustorl.common import *
from deustorl.montecarlo import Montecarlo_FirstVisit
from deustorl.helpers import FrozenLakeDenseRewardsWrapper

def train_and_evaluate(algo, n_steps= 60000, **kwargs):
    epsilon_greedy_policy = EpsilonGreedyPolicy(epsilon=0.1)
    start_time = time.time()
    algo.learn(epsilon_greedy_policy, n_steps, **kwargs)
    print("----- {:0.4f} secs. -----".format(time.time() - start_time))

    avg_reward, avg_steps = evaluate_policy(algo.env, algo.q_table, max_policy, n_episodes=100, verbose=False)
    print("Avg reward: {:0.4f}, avg steps: {:0.4f}".format(avg_reward,avg_steps))
    return avg_reward, avg_steps

os.system("rm -rf ./logs/")

n_rounds = 10
n_steps_per_round = 80_000

seed = 3
random.seed(seed)

map = generate_random_map(size=10, seed=seed)
for i in range(len(map)):
    print(map[i])
input("Press a key...")

env = gym.make("FrozenLake-v1", desc=map, is_slippery=False, render_mode="ansi")
env.reset(seed=seed)

print("Testing First Visit Montecarlo")
total_reward = 0
for i in range(n_rounds):
    algo = Montecarlo_FirstVisit(env)
    avg_reward, _ = train_and_evaluate(algo, n_steps=n_steps_per_round)
    total_reward += avg_reward

print("Avg reward: {:0.4f}".format(total_reward/n_rounds))
input("Press a key...")

env = FrozenLakeDenseRewardsWrapper(env)

print("Testing First Visit Montecarlo with Dense Rewards Wrapper")
total_reward = 0
for i in range(n_rounds):
    algo = Montecarlo_FirstVisit(env)
    avg_reward, _ = train_and_evaluate(algo, n_steps=n_steps_per_round)
    total_reward += avg_reward

print("Avg reward: {:0.4f}".format(total_reward/n_rounds))