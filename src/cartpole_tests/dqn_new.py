import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque

# Define the neural network architecture for the Q-network
class QNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# Define the Double DQN agent
class DoubleDQNAgent:
    def __init__(self, state_dim, action_dim, gamma=0.99, lr=0.001, batch_size=64, memory_size=10000, target_update_freq=10):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.lr = lr
        self.batch_size = batch_size
        self.memory = deque(maxlen=memory_size)
        self.target_update_freq = target_update_freq
        self.step_count = 0

        # Q-network and target network
        self.q_network = QNetwork(state_dim, action_dim)
        self.target_network = QNetwork(state_dim, action_dim)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

        # Exploration parameters
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995

    def choose_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_dim)
        else:
            state = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                q_values = self.q_network(state)
            return torch.argmax(q_values).item()

    def store_transition(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def update_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def train(self):
        if len(self.memory) < self.batch_size:
            return

        # Sample a batch from the replay memory
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions).unsqueeze(1)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones).unsqueeze(1)

        # Compute Q(s, a) using the main Q-network
        q_values = self.q_network(states).gather(1, actions)

        # Double DQN: Use the main network to select actions and the target network to compute Q-values
        with torch.no_grad():
            next_actions = self.q_network(next_states).argmax(dim=1, keepdim=True)
            q_targets_next = self.target_network(next_states).gather(1, next_actions)
            q_targets = rewards + (self.gamma * q_targets_next * (1 - dones))

        # Compute the loss and update the main Q-network
        loss = self.loss_fn(q_values, q_targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update the target network periodically
        self.step_count += 1
        if self.step_count % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

    def save(self, path):
        torch.save(self.q_network.state_dict(), path)

    def load(self, path):
        self.q_network.load_state_dict(torch.load(path))
        self.target_network.load_state_dict(self.q_network.state_dict())

def evaluate_agent(env_name, agent, num_episodes=10):
    env = gym.make(env_name)
    total_rewards = []

    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        done = False

        while not done:
            # Render the environment
            # env.render()

            # Agent takes an action using a greedy policy (without exploration)
            action = agent.choose_action(state)
            next_state, reward, done, _, _ = env.step(action)

            state = next_state
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

    agent = DoubleDQNAgent(state_dim, action_dim, batch_size=2)
    rewards = []
    evalrewards = []

    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        done = False

        while not done:
            action = agent.choose_action(state)
            next_state, reward, done, _, _ = env.step(action)
            agent.store_transition(state, action, reward, next_state, done)

            agent.train()
            state = next_state
            episode_reward += reward

            if done:
                agent.update_epsilon()
                rewards.append(episode_reward)
                print(f"Episode {episode + 1}/{num_episodes}, Reward: {episode_reward}, Epsilon: {agent.epsilon:.2f}")

        if episode != 0 and episode % 20 == 0:
            avg_reward = evaluate_agent(env_name, agent)
            evalrewards.append(avg_reward)
            print(f"Episode {episode + 1}/{num_episodes}, Average Reward: {avg_reward}")

    env.close()
    print("Training complete.")
    print("evalrewards: ", evalrewards)

    # save agent
    agent.save("chat_dqn.pth")
    
    return rewards

# Run the training
if __name__ == "__main__":
    rewards = train_agent("CartPole-v1", num_episodes=2)
