from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy
import os
import numpy as np
import matplotlib.pyplot as plt
from src.env import CallCenterEnv

# Create a directory to save logs
log_dir = "ppo_logs/"
os.makedirs(log_dir, exist_ok=True)

# Load environment with your dataset
env = DummyVecEnv([lambda: Monitor(CallCenterEnv("data/ModifiedCallCenter.xlsx"), log_dir)])

# Train or load model
model = PPO("MlpPolicy", env, verbose=1, learning_rate=1.04e-5,
            n_steps=1024,
            batch_size=128,
            gae_lambda=0.80,
            gamma=0.97,
            ent_coef=0.003)

# Train the model
model.learn(total_timesteps=100_000)

# Save model (optional)
model.save("ppo_call_center_model")

# Plot Loss Function (Training Rewards)
def plot_training_rewards(log_dir):
    """
    Plot the training rewards (loss function) from the Monitor logs.
    """
    x, y = ts2xy(load_results(log_dir), "timesteps")
    if len(x) > 0:
        plt.plot(x, y, label="Training Rewards")
        plt.xlabel("Timesteps")
        plt.ylabel("Rewards")
        plt.title("Training Rewards Over Time")
        plt.legend()
        plt.show()
    else:
        print("No training rewards found in the log directory.")

plot_training_rewards(log_dir)

# Evaluation
eval_env = CallCenterEnv("data/CallCenterDataset.xlsx")
obs, _ = eval_env.reset()  # Extract only the observation
done = False

total_reward = 0
episode_rewards = []
total_satisfaction = 0
agent_loads = np.zeros(eval_env.num_agents)
total_waiting_time = 0

while not done:
    action, _ = model.predict(obs)  # Use only the observation for prediction
    obs, reward, done, _, _ = eval_env.step(action)  # Extract observation and other values
    total_reward += reward
    agent_loads[action] += 1
    total_satisfaction += obs[-1]  # Last element in obs is satisfaction
    total_waiting_time += sum(obs[:eval_env.num_agents])  # Sum of queue lengths

# Metrics
avg_reward = total_reward
avg_satisfaction = total_satisfaction / eval_env.current_index
utilization = agent_loads / agent_loads.sum()
avg_waiting_time = total_waiting_time / eval_env.current_index

print("\n=== Evaluation Results ===")
print(f"Total reward: {total_reward:.2f}")
print(f"Average reward per call: {avg_reward / eval_env.current_index:.2f}")
print(f"Average satisfaction: {avg_satisfaction:.2f}")
print(f"Average waiting time: {avg_waiting_time:.2f}")
print("Agent utilization (fraction of calls handled):")
for i, frac in enumerate(utilization):
    print(f"  Agent {i}: {frac:.2%}")

# Plot Episode Rewards
plt.plot(eval_env.episode_rewards_list, label="Episode Rewards")
plt.xlabel("Episodes")
plt.ylabel("Total Reward")
plt.title("Episode Rewards Over Time")
plt.legend()
plt.show()