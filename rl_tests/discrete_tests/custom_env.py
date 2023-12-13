"""import sys

sys.path.append("gym-examples")
import gym_examples

from gymnasium.wrappers import FlattenObservation
import gymnasium as gym

env = gym.make("gym-examples/GridWorld-v0")"""
import gym_examples
import gymnasium
import numpy as np
import matplotlib.pyplot as plt
import pickle


def update_q_table(q_table, state, action, reward, next_state, alpha, gamma):
    # Convert states to indices suitable for accessing the Q-table
    # Assuming states are dictionaries with 'agent', 'target', and 'obstacles'
    agent_x, agent_y = state["agent"]
    target_x, target_y = state["target"]
    n, s, e, w = process_obstacles(state["agent"], state["obstacles"])

    next_agent_x, next_agent_y = next_state["agent"]
    next_target_x, next_target_y = next_state["target"]
    next_n, next_s, next_e, next_w = process_obstacles(
        next_state["agent"], next_state["obstacles"]
    )

    # Current Q value
    current_q = q_table[agent_x, agent_y, target_x, target_y, n, s, e, w, action]

    # Maximum Q value for the actions in the next state
    max_future_q = np.max(
        q_table[
            next_agent_x,
            next_agent_y,
            next_target_x,
            next_target_y,
            next_n,
            next_s,
            next_e,
            next_w,
        ]
    )

    # Q-learning formula
    new_q = current_q + alpha * (reward + gamma * max_future_q - current_q)

    # Update the Q-table
    q_table[agent_x, agent_y, target_x, target_y, n, s, e, w, action] = new_q


def process_obstacles(agent_location, obstacles, threshold=15):
    """
    Simplify obstacle information.

    Args:
    - agent_location: Tuple (x, y) for the location of the agent.
    - obstacles: List of tuples [(x1, y1), (x2, y2), ...] representing obstacle locations.
    - threshold: Distance within which an obstacle is considered 'close'.

    Returns:
    - Tuple of four binary values indicating the presence of an obstacle in the North, South, East, and West directions.
    """
    north_obstacle = south_obstacle = east_obstacle = west_obstacle = 0

    for obstacle in obstacles:
        # Calculate the difference in position
        dx, dy = obstacle[0] - agent_location[0], obstacle[1] - agent_location[1]

        # Check North (negative y-direction)
        if dy < 0 and abs(dy) <= threshold and abs(dx) <= threshold:
            north_obstacle = 1

        # Check South (positive y-direction)
        if dy > 0 and abs(dy) <= threshold and abs(dx) <= threshold:
            south_obstacle = 1

        # Check East (positive x-direction)
        if dx > 0 and abs(dx) <= threshold and abs(dy) <= threshold:
            east_obstacle = 1

        # Check West (negative x-direction)
        if dx < 0 and abs(dx) <= threshold and abs(dy) <= threshold:
            west_obstacle = 1

    return north_obstacle, south_obstacle, east_obstacle, west_obstacle


def run(episodes, is_training=False, render=False):
    env = gymnasium.make(
        "gym_examples/GridWorld-v0", render_mode="human" if render else None
    )

    print(env.observation_space)
    print(env.action_space)

    agent_grid = env.size  # 20x20 for agent position
    target_grid = env.size  # 20x20 for target position
    obstacle_representation = 2  # Binary flags for obstacles
    number_of_actions = 4  # Number of possible actions

    if is_training:
        # the four obstacle representations are for the 4 closest obstacles
        # in the north, south, east, west directions

        # Initialize Q-table with zeros
        q_table = np.zeros(
            (
                agent_grid,
                agent_grid,
                target_grid,
                target_grid,
                obstacle_representation,
                obstacle_representation,
                obstacle_representation,
                obstacle_representation,
                number_of_actions,
            )
        )

    else:
        f = open("2d_obstacle_q.pkl", "rb")
        q_table = pickle.load(f)
        f.close()

    # you can tune these
    learning_rate = 0.9
    discount_factor = 0.9
    epsilon = 1  # 1 = 100% random actions --> over time we will decrease the randomness by using the epsilon decay rate

    # epsilon decay rate. This number has a direct impact on how many epsiodes you need to train.
    epsilon_decay_rate = 0.000001  # 1/0.0001 = 10,000 to train b/c we subtract it from  1 after each epsiode
    rng = np.random.default_rng()  # random number generator

    # used to keep track of whether we collected a reward for an epsiode or not
    rewards_per_epsiode = np.zeros(episodes)

    # episodes = # of times we are going to train
    for ep in range(episodes):
        state = env.reset()[0]
        terminated = False  # True when fall in hole or reached goal
        # Basically, if the dude wanders around for more than 200 steps, the simulation ends.
        truncated = False  # True when actions > 200.

        while not terminated and not truncated:
            # SAMPLE ACTION --> Off policy

            # if we are training and
            # if the number we generate is less than epsilon, we take the random action
            # otherwise, we will follow the q table
            if is_training and rng.random() < epsilon:
                # Exploration: Choose a random action
                action = env.action_space.sample()

            else:
                # Exploitation: Choose the best action from Q-table
                # Convert state to indices suitable for accessing the Q-table
                # Assuming state is a dictionary with 'agent', 'target', and 'obstacles'
                agent_x, agent_y = state["agent"]
                target_x, target_y = state["target"]

                # Function to process obstacles into binary flags
                n, s, e, w = process_obstacles(state["agent"], state["obstacles"])

                q_values = q_table[agent_x, agent_y, target_x, target_y, n, s, e, w]
                action = np.argmax(q_values)

            # executing the actions returns the new state, the reward for doing that action, ...
            new_state, reward, terminated, truncated, _ = env.step(action)
            print(reward)
            if reward == 1000:
                raise
            # if it is training, update the q table
            if is_training:
                update_q_table(
                    q_table,
                    state,
                    action,
                    reward,
                    new_state,
                    learning_rate,
                    discount_factor,
                )
            state = new_state

            # after each episode, we decrease epsilon, all the way till it gets to zero.
            epsilon = max(epsilon - epsilon_decay_rate, 0)
            # we reduce the learning rate in order to help the q values stabilize after the system is done exploring
            if epsilon == 0:
                learning_rate = 0.0001

            rewards_per_epsiode[ep] = reward

    env.close()

    # get the rewards on a graph
    sum_rewards = np.zeros(episodes)
    # shows running sum of rewards for every 100 epsiodes
    for t in range(episodes):
        sum_rewards[t] = np.sum(rewards_per_epsiode[max(0, t - 100) : (t + 1)])
    plt.plot(sum_rewards)
    plt.savefig("2d_obstacle_q.png")

    # only if we are training, save the q table
    if is_training:
        f = open("2d_obstacle_q.pkl", "wb")
        pickle.dump(q_table, f)
        f.close()


if __name__ == "__main__":
    run(20000, is_training=True, render=True)
