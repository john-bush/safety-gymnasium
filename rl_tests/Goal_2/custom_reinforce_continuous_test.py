"""Goal level 0."""

# Introduce the required objects
from safety_gymnasium.assets.geoms import Goal

# Need to inherit from BaseTask
from safety_gymnasium.bases.base_task import BaseTask
import safety_gymnasium
import argparse
import gymnasium as gym
import random

import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from torch.distributions.normal import Normal

import gymnasium as gym
import torch.nn.functional as F
import torch.distributions as Categorical


class Policy_Network(nn.Module):
    """Parametrized Policy Network."""

    def __init__(self, obs_space_dims: int, action_space_dims: int):
        """Initializes a neural network that estimates the mean and standard deviation
         of a normal distribution from which an action is sampled from.

        Args:
            obs_space_dims: Dimension of the observation space
            action_space_dims: Dimension of the action space
        """
        super().__init__()

        hidden_space1 = 132  # Nothing special with 16, feel free to change
        hidden_space2 = 100  # Nothing special with 32, feel free to change
        hidden_space3 = 64  # Nothing special with 16, feel free to change
        hidden_space4 = 32  # Nothing special with 32, feel free to change

        # Shared Network
        self.shared_net = nn.Sequential(
            nn.Linear(obs_space_dims, hidden_space1),
            nn.Tanh(),
            nn.Linear(hidden_space1, hidden_space2),
            nn.Tanh(),
            nn.Linear(hidden_space2, hidden_space3),
            nn.Tanh(),
            nn.Linear(hidden_space3, hidden_space4),
            nn.Tanh(),
        )

        # Policy Mean specific Linear Layer
        self.policy_mean_net = nn.Sequential(nn.Linear(hidden_space4, action_space_dims))

        # Policy Std Dev specific Linear Layer
        self.policy_stddev_net = nn.Sequential(nn.Linear(hidden_space4, action_space_dims))

    def forward(self, x):
        """Conditioned on the observation, returns the mean and standard deviation
         of a normal distribution from which an action is sampled from.

        Args:
            x: Observation from the environment

        Returns:
            action_means: predicted mean of the normal distribution
            action_stddevs: predicted standard deviation of the normal distribution
        """
        x = torch.Tensor(x)
        shared_features = self.shared_net(x.float())

        action_means = self.policy_mean_net(shared_features)
        action_stddevs = torch.log(1 + torch.exp(self.policy_stddev_net(shared_features)))

        return (torch.Tensor(action_means), torch.Tensor(action_stddevs))


class REINFORCE:
    """REINFORCE algorithm."""

    def __init__(self, obs_space_dims: int, action_space_dims: int):
        """Initializes an agent that learns a policy via REINFORCE algorithm [1]
        to solve the task at hand (Inverted Pendulum v4).

        Args:
            obs_space_dims: Dimension of the observation space
            action_space_dims: Dimension of the action space
        """

        # Hyperparameters
        self.learning_rate = 1e-4  # Learning rate for policy optimization
        self.gamma = 0.99  # Discount factor
        self.eps = 1e-6  # small number for mathematical stability

        self.probs = []  # Stores probability values of the sampled action
        self.rewards = []  # Stores the corresponding rewards
        self.costs = []

        self.net = Policy_Network(obs_space_dims, action_space_dims)
        self.net.load_state_dict(torch.load("model_parameters.pth"))
        # comment this our later
        # self.net.load_state_dict(torch.load("model_parameters.pth"))
        self.optimizer = torch.optim.AdamW(self.net.parameters(), lr=self.learning_rate)

    def sample_action(self, state: np.ndarray) -> float:
        """Returns an action, conditioned on the policy and observation.

        Args:
            state: Observation from the environment

        Returns:
            action: Action to be performed
        """
        state = torch.tensor(np.array([state]))
        action_means, action_stddevs = self.net(state)

        # create a normal distribution from the predicted
        #   mean and standard deviation and sample an action
        distrib = Normal(action_means[0] + self.eps, action_stddevs[0] + self.eps)
        action = distrib.sample()
        prob = distrib.log_prob(action)

        action = action.numpy()

        self.probs.append(prob)

        return action

    '''def update(self):
        """Updates the policy network's weights."""
        running_g = 0
        gs = []

        # Discounted return (backwards) - [::-1] will return an array in reverse
        for R in self.rewards[::-1]:
            running_g = R + self.gamma * running_g
            gs.insert(0, running_g)

        deltas = torch.tensor(gs)

        loss = 0
        # minimize -1 * prob * reward obtained
        for log_prob, delta in zip(self.probs, deltas):
            loss += log_prob.mean() * delta * (-1)

        # Update the policy network
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Empty / zero out all episode-centric/related variables
        self.probs = []
        self.rewards = []'''

    def update(self):
        running_g = 0
        gs = []

        # Combine rewards and costs
        combined_rewards = [r - 0.5 * c for r, c in zip(self.rewards, self.costs)]

        # Discounted return (backwards)
        for R in combined_rewards[::-1]:
            running_g = R + self.gamma * running_g
            gs.insert(0, running_g)

        deltas = torch.tensor(gs)

        loss = 0
        # minimize -1 * prob * combined reward obtained
        for log_prob, delta in zip(self.probs, deltas):
            loss += log_prob.mean() * delta * (-1)

        # Update the policy network
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Empty / zero out all episode-centric/related variables
        self.probs = []
        self.rewards = []
        self.costs = []

    def save_model(self):
        print("saving model")
        torch.save(self.net.state_dict(), "model_parameters.pth")


def run_random(env_name):
    """Random run."""
    env = safety_gymnasium.make(env_name, render_mode="human")

    # This will keep track of cumulative rewards and episode lengths returning them at the end.
    # The most recent rewards and episode lengths are stored in buffers that can be accessed via wrapped_env.
    # return_queue and wrapped_env.length_queue respectively. They are of size 50 here
    """wrapped_env = gym.wrappers.RecordEpisodeStatistics(
        env, 50
    )  # Records episode-reward"""

    total_num_episodes = int(5e4)  # Total number of episodes

    obs_space_dims = env.observation_space.shape[0]
    action_space_dims = env.action_space.shape[0]
    rewards_over_seeds = []
    best_reward = -1 * float("inf")

    for seed in [1, 2, 3, 5, 8]:  # Fibonacci seeds
        # set seed
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)

        # Reinitialize agent every seed
        agent = REINFORCE(obs_space_dims, action_space_dims)
        reward_over_episodes = []
        reward_combined_over_episodes = []
        for episode in range(total_num_episodes):
            # gymnasium v26 requires users to set seed while resetting the environment
            obs, info = env.reset(seed=seed)
            ep_ret, ep_cost, ep_combo = 0, 0, 0
            done = False
            while not done:
                action = agent.sample_action(obs)
                action_2 = env.action_space.sample()

                # print(action)
                # print(action_2)
                # raise

                # Step return type - `tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]`
                # These represent the next observation, the reward from the step,
                # if the episode is terminated, if the episode is truncated and
                # additional info from the step
                obs, reward, cost, terminated, truncated, info = env.step(action)
                agent.rewards.append(reward)
                agent.costs.append(cost)

                # End the episode when either truncated or terminated is true
                #  - truncated: The episode duration reaches max number of timesteps
                #  - terminated: Any of the state space values is no longer finite.
                done = terminated or truncated
                ep_ret += reward
                ep_cost += cost
                ep_combo += reward - 0.5 * cost

            reward_over_episodes.append(ep_ret)
            reward_combined_over_episodes.append(ep_combo)
            # agent.update()
            if episode % 1000 == 0:
                avg_reward = int(np.mean(reward_combined_over_episodes))
                if avg_reward > best_reward:
                    # agent.save_model()
                    best_reward = avg_reward

                print("Episode:", episode, "Average Reward:", avg_reward)
    rewards_over_seeds.append(reward_combined_over_episodes)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="SafetyCarTestTask4-v0")
    args = parser.parse_args()
    run_random(args.env)
