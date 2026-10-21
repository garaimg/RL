import os
import gymnasium as gym
import random

from deustorl.common import *
from deustorl.montecarlo import Montecarlo_FirstVisit
from deustorl.montecarlo_lr import Montecarlo_FirstVisit_LR
from deustorl.montecarlo_lrdecay import Montecarlo_FirstVisit_LRDecay
from deustorl.sarsa import Sarsa

def train_and_evaluate(algo, n_steps= 60000, **kwargs):
    epsilon_greedy_policy = EpsilonGreedyPolicy(epsilon=0.1)
    start_time = time.time()
    algo.learn(epsilon_greedy_policy, n_steps, **kwargs)
    print("----- {:0.4f} secs. -----".format(time.time() - start_time))

    return evaluate_policy(algo.env, algo.q_table, max_policy, n_episodes=100, verbose=False)

os.system("rm -rf ./logs/")

env = gym.make("FrozenLake-v1", map_name="8x8", is_slippery=False, render_mode="ansi")

seed = 3
random.seed(seed)
env.reset(seed=seed)

n_rounds = 10
n_steps_per_round = 40_000

print("Testing First Visit Monte Carlo")
total_reward = 0
for _ in range(n_rounds):
    algo = Montecarlo_FirstVisit(env)
    avg_reward, _ = train_and_evaluate(algo, n_steps=n_steps_per_round)
    total_reward += avg_reward
print("------\r\nAverage reward over {} rounds: {:.4f}\r\n------".format(n_rounds, total_reward/n_rounds))

print("Testing First Visit Monte Carlo with Learning Rate and lrdecay")
total_reward = 0
for _ in range(n_rounds):
    algo = Montecarlo_FirstVisit_LRDecay(env)
    avg_reward, _ = train_and_evaluate(algo, n_steps=n_steps_per_round, lr=0.05, lrdecay=0.9, n_episodes_decay=50)
    total_reward += avg_reward
print("------\r\nAverage reward over {} rounds: {:.4f}\r\n------".format(n_rounds, total_reward/n_rounds))

print("Testing SARSA")
total_reward = 0
for _ in range(n_rounds):
    algo = Sarsa(env)
    avg_reward, _ = train_and_evaluate(algo, n_steps=n_steps_per_round)
    total_reward += avg_reward
print("------\r\nAverage reward over {} rounds: {:.4f}\r\n------".format(n_rounds, total_reward/n_rounds))

print("Testing SARSA with Learning Rate and lrdecay")
total_reward = 0
for _ in range(n_rounds):
    algo = Sarsa(env)
    avg_reward, _ = train_and_evaluate(algo, n_steps=n_steps_per_round, lr=0.05, lrdecay=0.9, n_episodes_decay=50)
    total_reward += avg_reward
print("------\r\nAverage reward over {} rounds: {:.4f}\r\n------".format(n_rounds, total_reward/n_rounds))

