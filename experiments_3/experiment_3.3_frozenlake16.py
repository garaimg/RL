import os
import sys
import gymnasium as gym
import random
from gymnasium.envs.toy_text.frozen_lake import generate_random_map
from deustorl.common import *
from deustorl.sarsa import Sarsa
from deustorl.Q_learning import QLearning
from deustorl.expected_sarsa import ExpectedSarsa
from deustorl.helpers import FrozenLakePotentialShapingWrapper

# FrozenLake-v1 16x16 (is_slippery=False) con un mapa aleatorio fijo. Recompensa +1 sólo al llegar a la meta.
# 1) Con la recompensa original (dispersa) el agente prácticamente nunca llega a la meta explorando al azar
#    (hacen falta >= 30 pasos sin caer en ningún agujero), así que la tabla Q se queda a 0.
# 2) Solución: reward shaping basado en potencial (FrozenLakePotentialShapingWrapper).
# La evaluación se hace SIEMPRE en el entorno original (eval_env): 1.0 = llega siempre a la meta.

def train_and_evaluate(algo, n_steps=200_000, epsilon=0.1, eps_decay=None, **kwargs):
    if eps_decay is None:
        policy = EpsilonGreedyPolicy(epsilon=epsilon)
    else:
        # Extra: decaying epsilon -> eps_decay = (epsilon inicial, epsilon mínimo, factor por episodio)
        policy = DecayingEpsilonGreedyPolicy(*eps_decay)
    start_time = time.time()
    algo.learn(policy, n_steps, **kwargs)
    print("----- {:0.4f} secs. -----".format(time.time() - start_time))

    return evaluate_policy(eval_env, algo.q_table, max_policy, n_episodes=100, verbose=False)


def test(algo_name, label, **kwargs):
    print("Testing {} [{}] {}".format(algo_name, label, kwargs))
    total_reward = 0
    for n in range(n_rounds):
        print("--- Round {}/{} ---".format(n + 1, n_rounds))
        algo = algorithms[algo_name](env)
        avg_reward, _ = train_and_evaluate(algo, n_steps=n_steps_per_round,
                                           log_dir=log_dir + label, log_name=algo_name + "_run" + str(n), **kwargs)
        total_reward += avg_reward

    print("Average reward over {} rounds: {:.4f}".format(n_rounds, total_reward/n_rounds))

    if show_agent:
        # Ver en una ventana cómo se comporta el último agente entrenado (política greedy)
        print("Showing the last trained agent ({} episodes)...".format(n_visual_episodes))
        evaluate_policy(visual_env, algo.q_table, max_policy, n_episodes=n_visual_episodes, verbose=False)
    results.append((log_dir, label, algo_name, total_reward/n_rounds))


random_map = generate_random_map(size=16, p=0.8, seed=42)
print("\n".join(random_map))

# El límite por defecto (100 pasos) es justo para un mapa 16x16
eval_env = gym.make("FrozenLake-v1", desc=random_map, is_slippery=False, max_episode_steps=200)

# Entorno visual para ver al agente después de cada prueba (como en experiment_2.9.py).
# Poner show_agent = False para ejecutar sin ventana (mucho más rápido).
show_agent = True
n_visual_episodes = 2
visual_env = gym.make("FrozenLake-v1", desc=random_map, is_slippery=False, render_mode="human", max_episode_steps=60)
visual_env.unwrapped.metadata = {**visual_env.unwrapped.metadata, "render_fps": 8}  # velocidad de la animación

seed = 3
random.seed(seed)
eval_env.reset(seed=seed)

n_rounds = 5
n_steps_per_round = 200_000

algorithms = {"sarsa": Sarsa, "qlearning": QLearning, "esarsa": ExpectedSarsa}
results = []

# ----- 1) Recompensa original (dispersa) -----
log_dir = "./logs/frozenlake16_sparse/"
os.system("rm -rf " + log_dir)

env = gym.make("FrozenLake-v1", desc=random_map, is_slippery=False, max_episode_steps=200)
env.reset(seed=seed)

for algo_name in algorithms:
    test(algo_name, "baseline", lr=0.1, discount_rate=0.99, epsilon=0.1)
    test(algo_name, "eps_decay", lr=0.1, discount_rate=0.99, eps_decay=(1.0, 0.01, 0.9995))

# ----- 2) Recompensa con reward shaping -----
log_dir = "./logs/frozenlake16_shaped/"
os.system("rm -rf " + log_dir)

env = FrozenLakePotentialShapingWrapper(gym.make("FrozenLake-v1", desc=random_map, is_slippery=False, max_episode_steps=200), gamma=0.99)
env.reset(seed=seed)

for algo_name in algorithms:
    # Baseline
    test(algo_name, "baseline", lr=0.1, discount_rate=0.99, epsilon=0.1)
    # Learning rate
    test(algo_name, "lr=0.01", lr=0.01, discount_rate=0.99, epsilon=0.1)
    test(algo_name, "lr=0.5", lr=0.5, discount_rate=0.99, epsilon=0.1)
    test(algo_name, "lr=0.5_decay", lr=0.5, lrdecay=0.95, n_episodes_decay=500, discount_rate=0.99, epsilon=0.1)
    # Discount rate
    test(algo_name, "dr=0.9", lr=0.1, discount_rate=0.9, epsilon=0.1)
    test(algo_name, "dr=1.0", lr=0.1, discount_rate=1.0, epsilon=0.1)
    # Epsilon
    test(algo_name, "eps=0.01", lr=0.1, discount_rate=0.99, epsilon=0.01)
    test(algo_name, "eps=0.3", lr=0.1, discount_rate=0.99, epsilon=0.3)
    # Extra: decaying epsilon (1.0 -> 0.01, x0.9995 por episodio; los episodios son cortos)
    test(algo_name, "eps_decay", lr=0.1, discount_rate=0.99, eps_decay=(1.0, 0.01, 0.9995))

print("\n===== Resumen (de mejor a peor) =====")
for log_dir, label, algo_name, avg_reward in sorted(results, key=lambda r: -r[3]):
    print("{:<28} {:<14} {:<10} {:9.4f}".format(log_dir, label, algo_name, avg_reward))

visual_env.close()
