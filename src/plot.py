import matplotlib.pyplot as plt

def plot_rewards(reward_history):
    plt.plot(reward_history)
    plt.xlabel("Episodes")
    plt.ylabel("Total Reward")
    plt.title("PPO Training Progress")
    plt.show()