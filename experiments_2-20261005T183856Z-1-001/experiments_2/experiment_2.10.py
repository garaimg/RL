import os
import gymnasium as gym
import random
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
import json

from gymnasium.envs.toy_text.frozen_lake import generate_random_map
from deustorl.common import *
from deustorl.sarsa import Sarsa
from deustorl.expected_sarsa import ExpectedSarsa
from deustorl.qlearning import QLearning
from deustorl.helpers import DiscretizedObservationWrapper

def objective(trial):

    # Normally, algorithm selection is not part of the hyperparameter optimization
    # but this is just to show the possibility and becaouse these 3 algorithms are very similar
    algo_name = trial.suggest_categorical("algo_name",["sarsa", "esarsa", "qlearning"])
    lr = trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True)
    lr_decay = trial.suggest_float("learning_rate_decay", 0.9 , 1.0,  step=0.01)
    lr_episodes_decay = trial.suggest_categorical("lr_episodes_decay",[100, 1_000, 10_000])
    discount_rate = trial.suggest_float("discount_rate", 0.8, 1.0, step=0.05)
    epsilon= trial.suggest_float("epsilon", 0.0, 0.4, step=0.05)
        	  	
    n_steps = 200_000

    if algo_name == "sarsa":
        algo = Sarsa(env)
    elif algo_name == "esarsa": 
        algo = ExpectedSarsa(env)
    elif algo_name == "qlearning": 
        algo = QLearning(env)

    epsilon_greedy_policy = EpsilonGreedyPolicy(epsilon=epsilon)
    algo.learn(epsilon_greedy_policy,n_steps=n_steps, discount_rate=discount_rate, lr=lr, lrdecay=lr_decay, n_episodes_decay=lr_episodes_decay)

    avg_reward, avg_steps = evaluate_policy(algo.env, algo.q_table, max_policy, n_episodes=100)
    
    return avg_reward
    
if __name__ == "__main__":

    # Create environment
    env_name = "CartPole-v1"
    env = DiscretizedObservationWrapper(gym.make(env_name), n_bins=10)
    seed = 47
    random.seed(seed)
    env.reset(seed=seed)

    # Remove previous logs and create optuna study directory if it does nto exist
    os.system("rm -rf ./logs/")
    os.system("mkdir -p ./optuna/")

    # Optuna study configuration
    storage_file = f"sqlite:///optuna/optuna.db"
    study_name = "cartpole"
    full_study_dir_path = f"optuna/{study_name}"
    tpe_sampler = TPESampler(seed=seed) # For reproducibility
    study = optuna.create_study(sampler=tpe_sampler, direction='maximize', study_name=study_name, storage=storage_file, load_if_exists=True)
    n_trials = 10 # Normally 50 or 100 at least

    # Start the study
    print(f"Searching for the best hyperparameters in {n_trials} trials...")
    study.optimize(objective, n_trials=n_trials)

    env.close()

    best_trial = study.best_trial

    # Generate the policy_kwargs key before writing to file
    best_trial_params = json.dumps(best_trial.params, sort_keys=True, indent=4)

    # Save the data in a JSON file
    os.system(f"mkdir -p {full_study_dir_path}")
    best_trial_file = open(f"{full_study_dir_path}/best_trial.json", "w")
    best_trial_file.write(best_trial_params)
    best_trial_file.close()

    # Generate the improtant figures of the results
    fig = optuna.visualization.plot_optimization_history(study)
    fig.write_html(f"{full_study_dir_path}/optimization_history.html")
    fig = optuna.visualization.plot_contour(study)
    fig.write_html(f"{full_study_dir_path}/contour.html")
    fig = optuna.visualization.plot_slice(study)
    fig.write_html(f"{full_study_dir_path}/slice.html")
    fig = optuna.visualization.plot_param_importances(study)
    fig.write_html(f"{full_study_dir_path}/param_importances.html")