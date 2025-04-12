import optuna
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from src.env import CallCenterEnv
import numpy as np

# Evaluation logic
def evaluate_model(model, eval_env, n_episodes=1):
    rewards = []
    for _ in range(n_episodes):
        obs = eval_env.reset()
        done = False
        total_reward = 0
        while not done:
            action, _ = model.predict(obs)
            obs, reward, done, _ = eval_env.step(action)
            total_reward += reward
        rewards.append(total_reward)
    return np.mean(rewards)

# Optuna objective function
def objective(trial):
    # Suggest hyperparameters
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-3)
    gamma = trial.suggest_uniform('gamma', 0.9, 0.999)
    gae_lambda = trial.suggest_uniform('gae_lambda', 0.8, 0.99)
    ent_coef = trial.suggest_loguniform('ent_coef', 0.00001, 0.01)
    n_steps = trial.suggest_categorical('n_steps', [512, 1024, 2048])
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])

    # Vectorized env
    train_env = make_vec_env(lambda: CallCenterEnv("data/ModifiedCallCenter.xlsx"), n_envs=1)
    model = PPO(
        "MlpPolicy", train_env,
        verbose=0,
        learning_rate=learning_rate,
        gamma=gamma,
        gae_lambda=gae_lambda,
        ent_coef=ent_coef,
        n_steps=n_steps,
        batch_size=batch_size,
    )

    model.learn(total_timesteps=50_000)

    # Evaluate
    eval_env = CallCenterEnv("data/ModifiedCallCenter.xlsx")
    mean_reward = evaluate_model(model, eval_env)

    # Report result to Optuna
    return mean_reward

# Run study
if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=20)

    print("Best trial:")
    trial = study.best_trial
    print(f"  Value: {trial.value}")
    print(f"  Params: {trial.params}")
