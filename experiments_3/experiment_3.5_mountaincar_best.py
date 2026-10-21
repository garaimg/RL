import os
import sys
import gymnasium as gym
import random
import json
from deustorl.common import *
from deustorl.sarsa import Sarsa
from deustorl.expected_sarsa import ExpectedSarsa
from deustorl.Q_learning import QLearning
from deustorl.helpers import DiscretizedObservationWrapper

# Entrenamiento largo en MountainCar-v0 con los mejores hiperparámetros encontrados por Optuna
# (experiment_3.4_mountaincar_optuna.py -> optuna/mountaincar/best_trial.json).
# Para comparar, se entrena también la configuración por defecto de experiment_2.9.py con los mismos pasos.

def train_and_evaluate(algo, policy, n_steps, **kwargs):
    start_time = time.time()
    algo.learn(policy, n_steps, **kwargs)
    print("----- {:0.4f} secs. -----".format(time.time() - start_time))

    return evaluate_policy(algo.env, algo.q_table, max_policy, n_episodes=100, verbose=False)


log_dir = "./logs/mountaincar_best/"
os.system("rm -rf " + log_dir)

env_name = "MountainCar-v0" # MountainCar-v0 best score is -110.0

with open("optuna/mountaincar/best_trial.json") as f:
    params = json.load(f)
print("Best hyperparameters found by Optuna:")
print(json.dumps(params, sort_keys=True, indent=4))

env = DiscretizedObservationWrapper(gym.make(env_name), n_bins=params["n_bins"])
visual_env = DiscretizedObservationWrapper(gym.make(env_name, render_mode='human'), n_bins=params["n_bins"])

seed = 3
random.seed(seed)
env.reset(seed=seed)

n_steps = 2_000_000  # Optuna usaba 300.000 pasos por trial
compare_with_default = True  # entrenar también la configuración de experiment_2.9.py
show_agent = True            # ver al agente en una ventana al terminar

algorithms = {"sarsa": Sarsa, "esarsa": ExpectedSarsa, "qlearning": QLearning}

print("Testing {} with the best hyperparameters ({} steps)".format(params["algo_name"], n_steps))
algo = algorithms[params["algo_name"]](env)
policy = DecayingEpsilonGreedyPolicy(epsilon=params["epsilon"], min_epsilon=0.0, decay=params["epsilon_decay"])
best_reward, _ = train_and_evaluate(algo, policy, n_steps,
                                    discount_rate=params["discount_rate"], lr=params["learning_rate"],
                                    lrdecay=params["learning_rate_decay"], n_episodes_decay=params["lr_episodes_decay"],
                                    log_dir=log_dir, log_name="best_" + params["algo_name"])
np.save("optuna/mountaincar/best_q_table.npy", algo.q_table.table)

if compare_with_default:
    print("Testing qlearning with the default hyperparameters of experiment_2.9.py ({} steps)".format(n_steps))
    default_env = DiscretizedObservationWrapper(gym.make(env_name), n_bins=10)
    default_env.reset(seed=seed)
    default_algo = QLearning(default_env)
    default_reward, _ = train_and_evaluate(default_algo, EpsilonGreedyPolicy(epsilon=0.1), n_steps, lr=0.1,
                                           log_dir=log_dir, log_name="default_qlearning")
    print("Average reward -> best: {:.2f}   default: {:.2f}".format(best_reward, default_reward))

if show_agent:
    print("Showing the agent trained with the best hyperparameters...")
    evaluate_policy(visual_env, algo.q_table, max_policy, n_episodes=5, verbose=False)
visual_env.close()
