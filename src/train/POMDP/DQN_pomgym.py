import numpy as np
import torch
import itertools
from tqdm import tqdm

from agents.DQN_agent import DoubleDQNAgent
from environment.env import POMDPGYMGridEnvDeform
from stable_baselines3.common.preprocessing import preprocess_obs

# maze size
N = 2

# thetas deformations (range(a,b),range(c,d))
l0 = 1
h0 = 10
l1 = 1
h1 = 10

maze = np.load(f"maze/maze_{N}.npy")
env = POMDPGYMGridEnvDeform(maze,l0,h0,l1,h1,render_mode='rgb_array')

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


def train_dqn(args):

    state_dim = len(env.deformations) + 3 # pos + belief over thetas
    action_dim = 4

    num_episodes = args.total_timesteps
    max_episode_steps = args.n_steps
    lr = args.learning_rate
    batch_size = args.batch_size

    agent = DoubleDQNAgent(state_dim, action_dim, lr = lr, batch_size=batch_size,target_update_freq=100, wandb=True)
    

    rewards = []
    evalrewards = []
    progress_bar = tqdm(total=num_episodes)

    for episode in range(num_episodes):
        progress_bar.set_description(f"episode {episode}")

        s, _ = env.reset()
        feedstate = torch.cat([ob for ob in s.values()])

        episode_reward = 0
        done = False
        steps = 0
        while not done and steps < max_episode_steps:
            

            action = agent.get_action(feedstate)
            
            s_ , reward, done , _, _ = env.step(action)
            nextfeedstate =  torch.cat([ob for ob in s_.values()])
            
            # state is now pos + beliefoverthetas

            agent.store_transition(feedstate, action, reward, nextfeedstate, done)

            agent.train()
          
            s = s_
            feedstate = nextfeedstate
          
            episode_reward += reward
            steps += 1      
        
        agent.update_epsilon()
        rewards.append(episode_reward)
                
        progress_bar.update(1)


        if episode != 0 and episode % 500 == 0:
            # avg_reward = evaluate_agent_training(env, agent)
            # evalrewards.append(avg_reward)
            # print(f"Episode {episode + 1}/{num_episodes}, Average Reward: {avg_reward}")
            pass
    print("Training complete.")
    agent.save("agents/double_dqn.pt")
    print("evalrewards: ", evalrewards)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser() 

    parser.add_argument("--learning_rate", type=float, default=0.0003)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--n_steps", type=int, default=200)
    parser.add_argument("--total_timesteps", type=int, default=50000)
        
    args = parser.parse_args()
    
    train_dqn(args)