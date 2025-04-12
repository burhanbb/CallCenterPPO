from stable_baselines3 import PPO
from env import CallCenterEnv

# Initialize environment
env = CallCenterEnv("data/ModifiedCallCenter.xlsx")

# Train PPO model
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10000)

# Save trained model
model.save("models/ppo_call_center")
