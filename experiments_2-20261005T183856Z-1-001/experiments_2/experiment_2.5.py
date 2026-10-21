import os
import gymnasium as gym
import random

from deustorl.common import *
from deustorl.montecarlo import Montecarlo_FirstVisit
from deustorl.montecarlo_lr import Montecarlo_FirstVisit_LR

def train_and_evaluate(algo, n_steps= 60000, **kwargs):
    epsilon_greedy_policy = EpsilonGreedyPolicy(epsilon=0.1)
    start_time = time.time()
    algo.learn(epsilon_greedy_policy, n_steps, **kwargs)
    print("----- {:0.4f} secs. -----".format(time.time() - start_time))

    evaluate_policy_by_steps(algo.env, algo.q_table, max_policy, 10_000, verbose=False)
    print()

os.system("rm -rf ./logs/")

env = gym.make("FrozenLake-v1", map_name="8x8", is_slippery=True, render_mode="ansi")

seed = 3
random.seed(seed)
env.reset(seed=seed)

n_steps = 400_000

algo = Montecarlo_FirstVisit(env)
print("Testing First Visit Monte Carlo")
train_and_evaluate(algo, n_steps=n_steps, tb_episode_period=1000)

algo = Montecarlo_FirstVisit_LR(env)
print("Testing First Visit Monte Carlo with Learning Rate 0.1")
train_and_evaluate(algo, n_steps=n_steps, lr=0.1, tb_episode_period=1000)

algo = Montecarlo_FirstVisit_LR(env)
print("Testing First Visit Monte Carlo with Learning Rate 0.01")
train_and_evaluate(algo, n_steps=n_steps, lr=0.01, tb_episode_period=1000)

algo = Montecarlo_FirstVisit_LR(env)
print("Testing First Visit Monte Carlo with Learning Rate 0.001")
train_and_evaluate(algo, n_steps=n_steps, lr=0.001, tb_episode_period=1000)

print(algo.q_table)

