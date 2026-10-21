import os
import sys
import gymnasium as gym
import random
import optuna
from optuna.samplers import TPESampler
import json
from deustorl.common import *
from deustorl.sarsa import Sarsa
from deustorl.expected_sarsa import ExpectedSarsa
from deustorl.Q_learning import QLearning
from deustorl.helpers import DiscretizedObservationWrapper

# MountainCar-v0: -1 por paso, episodios truncados a 200 pasos (-200 = no llega a la meta).
# La mejor puntuación conocida está alrededor de -110.

def objective(trial):

    # Normally, algorithm selection is not part of the hyperparameter optimization
    # but this is just to show the possibility and because these 3 algorithms are very similar
    algo_name = trial.suggest_categorical("algo_name", ["sarsa", "esarsa", "qlearning"])
    n_bins = trial.suggest_categorical("n_bins", [10, 20, 30, 40])
    lr = trial.suggest_float("learning_rate", 1e-3, 1.0, log=True)
    lr_decay = trial.suggest_float("learning_rate_decay", 0.9, 1.0, step=0.01)
    lr_episodes_decay = trial.suggest_categorical("lr_episodes_decay", [100, 1_000, 10_000])
    discount_rate = trial.suggest_float("discount_rate", 0.9, 1.0, step=0.01)
    epsilon = trial.suggest_float("epsilon", 0.0, 0.4, step=0.05)
    # Decaying epsilon: factor por episodio (1.0 = epsilon constante)
    epsilon_decay = trial.suggest_categorical("epsilon_decay", [1.0, 0.999, 0.995, 0.99])

    env = DiscretizedObservationWrapper(gym.make(env_name), n_bins=n_bins)
    env.reset(seed=seed + trial.number)

    if algo_name == "sarsa":
        algo = Sarsa(env)
    elif algo_name == "esarsa":
        algo = ExpectedSarsa(env)
    elif algo_name == "qlearning":
        algo = QLearning(env)

    policy = DecayingEpsilonGreedyPolicy(epsilon=epsilon, min_epsilon=0.0, decay=epsilon_decay)
    algo.learn(policy, n_steps=n_steps, discount_rate=discount_rate, lr=lr, lrdecay=lr_decay, n_episodes_decay=lr_episodes_decay,
               log_dir=log_dir, log_name="trial" + str(trial.number) + "_" + algo_name)

    avg_reward, avg_steps = evaluate_policy(algo.env, algo.q_table, max_policy, n_episodes=100)

    return avg_reward

if __name__ == "__main__":

    env_name = "MountainCar-v0"
    seed = 47
    random.seed(seed)

    n_steps = 300_000  # pasos de entrenamiento en cada trial
    n_trials = 50      # Normally 50 or 100 at least

    # Remove previous logs and create optuna study directory if it does not exist
    log_dir = "./logs/mountaincar_optuna/"
    os.system("rm -rf " + log_dir)
    os.system("mkdir -p ./optuna/")

    # Optuna study configuration
    storage_file = f"sqlite:///optuna/optuna.db"
    study_name = "mountaincar"
    full_study_dir_path = f"optuna/{study_name}"
    tpe_sampler = TPESampler(seed=seed) # For reproducibility
    study = optuna.create_study(sampler=tpe_sampler, direction='maximize', study_name=study_name, storage=storage_file, load_if_exists=True)

    # Start the study
    print(f"Searching for the best hyperparameters in {n_trials} trials...")
    study.optimize(objective, n_trials=n_trials)

    best_trial = study.best_trial
    print("Best trial: #{} with average reward {:.2f}".format(best_trial.number, best_trial.value))
    print(json.dumps(best_trial.params, sort_keys=True, indent=4))

    # Save the data in a JSON file
    os.system(f"mkdir -p {full_study_dir_path}")
    best_trial_file = open(f"{full_study_dir_path}/best_trial.json", "w")
    best_trial_file.write(json.dumps(best_trial.params, sort_keys=True, indent=4))
    best_trial_file.close()

    # Generate the important figures of the results
    fig = optuna.visualization.plot_optimization_history(study)
    fig.write_html(f"{full_study_dir_path}/optimization_history.html")
    fig = optuna.visualization.plot_contour(study)
    fig.write_html(f"{full_study_dir_path}/contour.html")
    fig = optuna.visualization.plot_slice(study)
    fig.write_html(f"{full_study_dir_path}/slice.html")
    fig = optuna.visualization.plot_param_importances(study)
    fig.write_html(f"{full_study_dir_path}/param_importances.html")
