import numpy as np
from stable_baselines3 import PPO
from env import CallCenterEnv
from plot import plot_rewards

def evaluate_model(env, model, episodes=100):
    rewards = []
    for _ in range(episodes):
        obs = env.reset()
        done = False
        episode_reward = 0
        while not done:
            action, _ = model.predict(obs)
            obs, reward, done, _ = env.step(action)
            episode_reward += reward
        rewards.append(episode_reward)
    
    print(f"Average reward: {np.mean(rewards)}")
    return np.mean(rewards)

env = CallCenterEnv("data/ModifiedCallCenter.xlsx")
model = PPO.load("models/ppo_call_center")

reward_history = evaluate_model(env, model)

plot_rewards(reward_history)
