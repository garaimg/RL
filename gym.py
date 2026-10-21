import gymnasium as gym
env = gym.make('CartPole-v1', render_mode="human")

for _ in range(100): # episodes
    obs, info = env.reset()
    for _ in range(1000): # steps per episode
        obs, reward, terminated, truncated, info  = env.step(env.action_space.sample()) # take a random action
        if terminated or truncated:
            break
env.close()
