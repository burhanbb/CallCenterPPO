import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd

class CallCenterEnv(gym.Env):
    def __init__(self, excel_path):
        super(CallCenterEnv, self).__init__()
        self.df = pd.read_excel(excel_path)
        self.df = self.df.dropna()
        self.df = self.df.reset_index(drop=True)
        self.current_index = 0
        self.num_agents = 9

        # Encode unique topics as customer types
        self.topics = self.df['Topic'].astype(str).unique().tolist()
        self.topic_to_id = {topic: i for i, topic in enumerate(self.topics)}

        # Observation space: 9 queue sizes + 1 customer type (topic ID) + 1 satisfaction rating
        self.observation_space = spaces.Box(
            low=0, high=np.inf, shape=(self.num_agents + 2,), dtype=np.float32
        )

        # Action space: choose an agent (0 to 8)
        self.action_space = spaces.Discrete(self.num_agents)

        # Service rates (can be customized per agent)
        self.service_rates = [1.0] * self.num_agents

        # Track episode rewards
        self.episode_rewards = 0
        self.episode_rewards_list = []

        self.reset()

    def reset(self, seed=None, options=None):
        """
        Reset the environment to its initial state.
        """
        super().reset(seed=seed)
        self.current_index = 0
        self.state = [0] * self.num_agents + [0, 5]  # Initial state: queues + customer_type + satisfaction

        # Store the total reward for the last episode
        if self.episode_rewards > 0:
            self.episode_rewards_list.append(self.episode_rewards)

        # Reset episode rewards
        self.episode_rewards = 0

        return np.array(self.state, dtype=np.float32), {}

    def seed(self, seed=None):
        """
        Set the random seed for reproducibility.
        """
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def expected_cost(self, queue_lengths, assigned_agent, customer_type):
        """
        Calculate expected cost based on the agent assignment and customer type.
        """
        updated_queues = queue_lengths.copy()
        updated_queues[assigned_agent] += 1

        if customer_type % 2 == 0:
            return sum([
                updated_queues[i] / self.service_rates[i]
                for i in range(self.num_agents)
            ])
        else:
            return sum([
                updated_queues[i] / (self.service_rates[i] + 0.1)
                for i in range(self.num_agents)
            ])

    def step(self, action):
        """
        Take a step in the environment.
        """
        if self.current_index >= len(self.df):
            return np.array(self.state, dtype=np.float32), 0.0, True, False, {}

        row = self.df.iloc[self.current_index]
        topic = str(row['Topic'])
        customer_type = self.topic_to_id.get(topic, 0)  # Fallback to 0 if unseen
        satisfaction = row['Satisfaction rating']

        agent_id = action
        queue_lengths = self.state[:self.num_agents]

        # Dynamically adjust service rates based on queue lengths
        self.service_rates = [1.0 / (1 + queue) for queue in queue_lengths]

        # Cost and reward calculation using expected cost
        cost = self.expected_cost(queue_lengths, agent_id, customer_type)

        # Adjust penalties and weights
        load_penalty = np.std(queue_lengths)
        usage_bonus = 0.1 * (1 - queue_lengths[action] / (sum(queue_lengths) + 1))
        reward = satisfaction - np.log(1 + cost) - 0.05 * sum(queue_lengths) - 0.1 * load_penalty + usage_bonus

        # Add positive rewards
        if satisfaction > 4:  # High satisfaction bonus
            reward += 5
        if queue_lengths[agent_id] == 0:  # Bonus for serving immediately
            reward += 2

        # Add a time-based reward to encourage faster service
        time_bonus = max(0, 10 - self.current_index)  # Bonus decreases as time progresses
        reward += time_bonus

        # Add priority for certain customer types
        priority_multiplier = 1.5 if customer_type in [0, 1] else 1.0  # Prioritize certain customer types
        reward *= priority_multiplier

        # Update queue for the selected agent
        self.state[agent_id] += 1

        # Update other state components
        self.state[-2] = customer_type
        self.state[-1] = satisfaction

        # Update episode rewards
        self.episode_rewards += reward

        self.current_index += 1
        done = self.current_index >= len(self.df)

        return np.array(self.state, dtype=np.float32), reward, done, False, {}

    def render(self, mode='human'):
        print(f"Step {self.current_index}, State: {self.state}")
