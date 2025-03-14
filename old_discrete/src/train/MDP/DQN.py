import numpy as np
import torch
import itertools
from tqdm import tqdm

from agents.DQN_agent import DoubleDQNAgent
from environment.env import GridEnvDeform
import wandb

# maze size
N = 2

# thetas deformations (range(a,b),range(c,d))
l0 = 1
h0 = 10
l1 = 1
h1 = 10

maze = np.load(f"maze/maze_{N}.npy")
env = GridEnvDeform(maze,l0,h0,l1,h1, render_mode="rgb_array")

states = [((x,y,phi),(i,j)) for x in range(1,env.max_shape[0]-1) for y in range(1,env.max_shape[1]-1) for phi in range(4) for i in range(l0,h0) for j in range(l1,h1)] 
actions = [0,1,2,3]
obs = list(itertools.product([0,1], repeat=5))
thetas = [(i,j) for i in range(l0,h0) for j in range(l1,h1)]

state_dict = {state: i for i, state in enumerate(states)}
obs_dict = {obs : i for i, obs in enumerate(obs)}

# Actions are: 0-listen, 1-open-left, 2-open-right
lenS = len(states)
lenA = len(actions)
lenO = len(obs)

print(f"States: {lenS}, Actions: {lenA}, Observations {lenO}, Thetas {thetas}\n")



def evaluate_agent_training(env : GridEnvDeform, agent : DoubleDQNAgent, num_episodes=10):
    total_rewards = []

    for episode in range(num_episodes):
        s, _ = env.reset()
        state = torch.tensor([item for sublist in s for item in sublist], dtype=torch.float32)

        episode_reward = 0
        done = False
        c = 25
        while not done and c > 0:
            # Render the environment
            # env.render()

            # Agent takes an action using a greedy policy (without exploration)
            action = agent.choose_deterministic_action(state)
            next_state, reward, done, _, info = env.step(action,s)

            state = torch.tensor([item for sublist in next_state for item in sublist], dtype=torch.float32)
            s = next_state

            episode_reward += reward
            
            if done or c == 1:
                total_rewards.append(episode_reward)
                # print(f"Episode {episode + 1}/{num_episodes}, Reward: {episode_reward}")

            c -= 1
    avg_reward = np.mean(total_rewards)
    # print(f"Average Reward over {num_episodes} episodes: {avg_reward}")
    return avg_reward



def train_dqn(args):

    state_dim = 5
    action_dim = 4

    num_episodes = args.num_episodes
    max_episode_steps = args.n_steps
    lr = args.learning_rate
    batch_size = args.batch_size
    gamma = args.gamma
    target_update = args.target_update

    config = {
        "num_episodes": num_episodes,
        "max_episode_steps": max_episode_steps,
        "lr": lr,
        "gamma": args.gamma,
        "batch_size": batch_size,
        "target_update_freq": target_update,
    }

    agent = DoubleDQNAgent(state_dim, action_dim, lr = lr,gamma=gamma, 
                           batch_size=batch_size,target_update_freq=target_update,
                             wandb=True, project_name="DQN - MDP", config=config)

    

    rewards = []
    evalrewards = []
    progress_bar = tqdm(total=num_episodes)
    total_steps = 0
    agent_trained = 0
    for episode in range(num_episodes):
        progress_bar.set_description(f"episode {episode}")

        s, _ = env.reset()
        state = torch.tensor([item for sublist in s for item in sublist], dtype=torch.float32)

        episode_reward = 0
        done = False
        
        steps = 0
        while not done and steps < max_episode_steps:
            action = agent.get_action(state)
            
            s_ , reward, done , _, _ = env.step(action,s, execute=True)
            next_state = torch.tensor([item for sublist in s_ for item in sublist], dtype=torch.float32)
            s = s_

            agent.store_transition(state, action, reward, next_state, done)

            
            agent.train()
            agent_trained += 1
            if agent_trained % 10 == 0:
                wandb.log({"env_steps":total_steps})

            state = next_state
            episode_reward += reward
            steps += 1    
            total_steps += 1  
        
        agent.update_epsilon()
        rewards.append(episode_reward)
                
        progress_bar.update(1)


        if episode != 0 and episode % 100 == 0:
            # avg_reward = evaluate_agent_training(env, agent)
            # evalrewards.append(avg_reward)
            print(f"Episode {episode}, saving checkpoint")
            agent.save(f"agents/pretrained/MDP/double_dqn_{episode}.pt")
            print("-- Saved\n")

    print("Training complete.")
    print("env stepped: ", total_steps)
    agent.save("agents/pretrained/MDP/double_dqn.pt")
    print("evalrewards: ", evalrewards)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser() 

    parser.add_argument("--learning_rate", type=float, default=0.0003)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--n_steps", type=int, default=200) # steps per episode --> num steps total = n_steps * num_episodes = 10000000 * 200
    parser.add_argument("--num_episodes", type=int, default=50000) 
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--target_update", type=int, default=100)

    args = parser.parse_args()
    
    train_dqn(args)