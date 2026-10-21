import os
import sys
import gymnasium as gym
import random
from deustorl.common import *
from deustorl.sarsa import Sarsa
from deustorl.Q_learning import QLearning
from deustorl.expected_sarsa import ExpectedSarsa

# CliffWalking-v1: rejilla 4x12 (48 estados), 4 acciones. -1 por paso y -100 al caer por el acantilado
# (vuelve al inicio sin terminar el episodio). El camino óptimo (pegado al acantilado) da -13.
# El entorno no tiene límite de pasos: añadimos uno (500) para que evaluar una política greedy
# que se queda en bucle no sea infinito.

def train_and_evaluate(algo, n_steps=200_000, epsilon=0.1, eps_decay=None, **kwargs):
    if eps_decay is None:
        policy = EpsilonGreedyPolicy(epsilon=epsilon)
    else:
        # Extra: decaying epsilon -> eps_decay = (epsilon inicial, epsilon mínimo, factor por episodio)
        policy = DecayingEpsilonGreedyPolicy(*eps_decay)
    start_time = time.time()
    algo.learn(policy, n_steps, **kwargs)
    print("----- {:0.4f} secs. -----".format(time.time() - start_time))

    return evaluate_policy(algo.env, algo.q_table, max_policy, n_episodes=100, verbose=False)


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
    results.append((label, algo_name, total_reward/n_rounds))


log_dir = "./logs/cliffwalking/"
os.system("rm -rf " + log_dir)

env = gym.make("CliffWalking-v1", max_episode_steps=500)

# Entorno visual para ver al agente después de cada prueba (como en experiment_2.9.py).
# Poner show_agent = False para ejecutar sin ventana (mucho más rápido).
show_agent = True
n_visual_episodes = 2
visual_env = gym.make("CliffWalking-v1", render_mode="human", max_episode_steps=50)
visual_env.unwrapped.metadata = {**visual_env.unwrapped.metadata, "render_fps": 8}  # velocidad de la animación

seed = 3
random.seed(seed)
env.reset(seed=seed)

n_rounds = 5
n_steps_per_round = 200_000

algorithms = {"sarsa": Sarsa, "qlearning": QLearning, "esarsa": ExpectedSarsa}
results = []

for algo_name in algorithms:
    # Baseline
    test(algo_name, "baseline", lr=0.1, discount_rate=0.99, epsilon=0.1)
    # Learning rate
    test(algo_name, "lr=0.01", lr=0.01, discount_rate=0.99, epsilon=0.1)
    test(algo_name, "lr=0.5", lr=0.5, discount_rate=0.99, epsilon=0.1)
    test(algo_name, "lr=0.5_decay", lr=0.5, lrdecay=0.95, n_episodes_decay=100, discount_rate=0.99, epsilon=0.1)
    # Discount rate
    test(algo_name, "dr=0.9", lr=0.1, discount_rate=0.9, epsilon=0.1)
    test(algo_name, "dr=1.0", lr=0.1, discount_rate=1.0, epsilon=0.1)
    # Epsilon
    test(algo_name, "eps=0.01", lr=0.1, discount_rate=0.99, epsilon=0.01)
    test(algo_name, "eps=0.3", lr=0.1, discount_rate=0.99, epsilon=0.3)
    # Extra: decaying epsilon (1.0 -> 0.01, x0.99 por episodio)
    test(algo_name, "eps_decay", lr=0.1, discount_rate=0.99, eps_decay=(1.0, 0.01, 0.99))

print("\n===== Resumen (de mejor a peor) =====")
for label, algo_name, avg_reward in sorted(results, key=lambda r: -r[2]):
    print("{:<14} {:<10} {:9.4f}".format(label, algo_name, avg_reward))

visual_env.close()
