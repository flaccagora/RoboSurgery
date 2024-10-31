import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque

import sys
import os

# Add the parent directory to sys.path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

from old.dqn_old import DQNAgent

def evaluate_agent(env_name, agent, num_episodes=10):
    env = gym.make(env_name)
    total_rewards = []
    la =  {i:i for i in range(2)}

    for episode in range(num_episodes):
        state, _ = env.reset()
        s = {'obs':torch.tensor(state), 'raw_legal_actions':la, 'legal_actions':la}

        episode_reward = 0
        done = False

        while not done:
            # Render the environment
            # env.render()

            # Agent takes an action using a greedy policy (without exploration)
            action = agent.step(s)
            next_state, reward, done, _, _ = env.step(action)

            state = next_state
            s = {'obs':torch.tensor(state), 'raw_legal_actions':la, 'legal_actions':la}

            episode_reward += reward

            if done:
                total_rewards.append(episode_reward)
                print(f"Episode {episode + 1}/{num_episodes}, Reward: {episode_reward}")

    env.close()
    avg_reward = np.mean(total_rewards)
    print(f"Average Reward over {num_episodes} episodes: {avg_reward}")
    return avg_reward

# Training loop
def train_agent(env_name, num_episodes=500):
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    print(action_dim)
    la =  {i:i for i in range(action_dim)}


    agent = DQNAgent(state_shape=(state_dim,), device="cpu",
                      mlp_layers=[128, 128], num_actions=action_dim,
                      learning_rate=0.001, discount_factor=0.99,
                      update_target_estimator_every=10,
                      replay_memory_size=10000,
                      epsilon_start=1.0, epsilon_end=0.01,epsilon_decay_steps=500,)
    rewards = []
    evalrewards = []

    for episode in range(num_episodes):
        state, _ = env.reset()
        s = {'obs':torch.tensor(state), 'raw_legal_actions':la, 'legal_actions':la}
        episode_reward = 0
        done = False

        while not done:
            action = agent.step(s)
            next_state, reward, done, _, _ = env.step(action)
            # agent.store_transition(state, action, reward, next_state, done)
            s_ = {'obs':torch.tensor(next_state), 'raw_legal_actions':la, 'legal_actions':la}
            agent.feed((s, action, reward, s_, done))

            state = next_state
            episode_reward += reward

            if done:
                rewards.append(episode_reward)
                print(f"Episode {episode + 1}/{num_episodes}, Reward: {episode_reward}")

        if episode != 0 and episode % 20 == 0:
            avg_reward = evaluate_agent(env_name, agent)
            evalrewards.append(avg_reward)
            print(f"Episode {episode + 1}/{num_episodes}, Average Reward: {avg_reward}")

    env.close()
    print("Training complete.")
    print("evalrewards: ", evalrewards)

    
    return rewards

# Run the training
if __name__ == "__main__":
    rewards = train_agent("CartPole-v1", num_episodes=500)
