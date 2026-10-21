import sys
import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

if len(sys.argv) < 2:
    print("Please, provide the number of steps to train as argument")
    print("Example: python " + sys.argv[0] + " 1000")
    quit()
else:
    n_timesteps = sys.argv[1]

# Create environment
env = gym.make('CartPole-v1', render_mode=None)

# Instantiate the agent
model = PPO('MlpPolicy', env, learning_rate=1e-3, verbose=1)

# Train the agent
print("Training for %s timesteps..."%(n_timesteps))
model.learn(total_timesteps=int(n_timesteps))

# Evaluate the agent
mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)
print("Mean reward after training: " + str(mean_reward))

env = gym.make('CartPole-v1', render_mode="human")

# Enjoy trained agent
for _ in range(10): # episodes
    obs, info = env.reset()
    for i in range(1000): #steps
        action, _state = model.predict(obs)
        obs, rewards, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break
