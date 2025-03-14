from tqdm import trange
import json

from stable_baselines3 import PPO, DQN
from environment.env import Grid
from utils.checkpoints import find_last_checkpoint
from environment.env import POMDPDeformedGridworld
from collections import OrderedDict
import torch
import numpy as np
import time
from agents.agent import TS

N_EPISODES = 1000
BELIEFUPDATE = 'discrete'
DISCRETIZATION = 10


def load_obs_model(obs_type):
    from observation_model.obs_model import singleNN, cardinalNN

    if obs_type == 'single':
        obs_model = singleNN()
        obs_model.load_state_dict(torch.load("observation_model/obs_model_4.pth", weights_only=True,map_location=torch.device('cpu')))
    elif obs_type == 'cardinal':
        obs_model = cardinalNN()
        obs_model.load_state_dict(torch.load("observation_model/obs_model_cardinal_4.pth", weights_only=True))
    else:
        raise ValueError("Observation type not recognized")
    
    return obs_model

def evaluate_statistics(transitions):
    """
    Evaluate statistics from transitions data organized as transitions[episode][step].
    
    Parameters:
    transitions: list of lists, where transitions[episode][step] contains
                a tuple (s, best_action, reward, next_state, terminated, truncated)
    
    Returns:
    dict with statistics including:
    - mean_episode_reward: Mean reward per episode
    - std_episode_reward: Standard deviation of rewards per episode
    - mean_episode_steps: Mean number of steps per episode
    - std_episode_steps: Standard deviation of steps per episode
    - terminated_episodes: Number of episodes that terminated naturally
    - truncated_episodes: Number of episodes that were truncated
    """
    episode_rewards = []
    episode_steps = []
    terminated_count = 0
    truncated_count = 0
    
    # Process each episode
    for episode_transitions in transitions:
        total_reward = 0
        num_steps = len(episode_transitions)
        
        # Check the final transition to determine if episode terminated or truncated
        if num_steps > 0:
            final_transition = episode_transitions[-1]
            s, action, reward, next_s, terminated, truncated = final_transition
            
            if terminated:
                terminated_count += 1
            if truncated:
                truncated_count += 1
        
        # Calculate total reward for the episode
        for step in range(num_steps):
            s, action, reward, next_s, terminated, truncated = episode_transitions[step]
            total_reward += reward
        
        # Record stats for this episode
        episode_rewards.append(total_reward)
        episode_steps.append(num_steps)
    
    # Calculate statistics
    stats = {
        'mean_episode_reward': np.mean(episode_rewards) if episode_rewards else 0,
        'std_episode_reward': np.std(episode_rewards) if episode_rewards else 0,
        'mean_episode_steps': np.mean(episode_steps) if episode_steps else 0,
        'std_episode_steps': np.std(episode_steps) if episode_steps else 0,
        'num_episodes': len(episode_rewards),
        'terminated_episodes': terminated_count,
        'truncated_episodes': truncated_count,
        'termination_rate': terminated_count / len(episode_rewards) if episode_rewards else 0,
        'truncation_rate': truncated_count / len(episode_rewards) if episode_rewards else 0
    }
    
    return stats



env = Grid(
    shear_range=(-.2, .2),
    stretch_range=(.4,1),
    render_mode="rgb_array"
)

run = "DQN_continous_" + "e2qthdat"

last_checkpoint = find_last_checkpoint(f"agents/pretrained/MDP/{run}")
DQN_model = DQN.load(f"agents/pretrained/MDP/{run}/{last_checkpoint}", env=env)

env.close()

#  for the best POMDP agent (TS, MLS, QMDP) and the best MDP agent (PPO, DQN)
# compute T_POMDP(x_0) / T_MDP(x_0) for each x_0 in the set of initial positions
# compute the mean and std of the above ratio

# MDP eval using DQN

import pandas as pd
data = pd.DataFrame(columns=['agent', 'model', 'obs_type', 'update', 'discretization', 'mean_episode_reward', 'std_episode_reward', 'mean_episode_steps', 'std_episode_steps', 'num_episodes', 'terminated_episodes', 'truncated_episodes', 'termination_rate', 'truncation_rate'])


def eval_agent_mdp(agent,env,num_episodes):
    """Returns
        - episode_transition: list of list of tuples (s,a,r,s',done), t[i] is the ith episode
        - beliefs: list of beliefs at each time step 
    """
    transitions = []
    total_steps = []
    for i in trange(num_episodes):
        s, _ = env.reset()

        totalReward = 0.0
        done = False
        steps = 0

        ep_transitions = []

        while not done:
            best_action, _ = agent.predict(s,deterministic=True)
            next_state, reward, terminated, truncated, info = env.step(best_action)
            
            # torch_dict = {key: torch.tensor(val, dtype=torch.float32).unsqueeze(0).to(agent.device) for key, val in next_state.items()}
            # value = agent.policy.predict_values(torch_dict)        
            # print(value)
            totalReward += reward            

            done = terminated or truncated
            s = next_state
            steps += 1

            ep_transitions.append((s, best_action, reward, next_state, terminated, truncated))
            # time.sleep(0.05)
    
        transitions.append(ep_transitions)
        total_steps.append(steps)

    env.close()

    return transitions, total_steps

env = Grid(
    render_mode="rgb_array",
)

MDP_AGENT = DQN_model
mdp_transitions, mdp_steps = eval_agent_mdp(DQN_model,env,N_EPISODES)

stats = evaluate_statistics(mdp_transitions)
# insert stats into dataframe
data = data._append({
    'agent': MDP_AGENT.__class__.__name__,
    'model': None,
    'obs_type': None,
    'update': None,
    'discretization': None,
    'mean_episode_reward': stats['mean_episode_reward'],
    'std_episode_reward': stats['std_episode_reward'],
    'mean_episode_steps': stats['mean_episode_steps'],
    'std_episode_steps': stats['std_episode_steps'],
    'num_episodes': stats['num_episodes'],
    'terminated_episodes': stats['terminated_episodes'],
    'truncated_episodes': stats['truncated_episodes'],
    'termination_rate': stats['termination_rate'],
    'truncation_rate': stats['truncation_rate']
}, ignore_index=True)

# save dataframe to file
data.to_csv("results.csv", index=False)

print(json.dumps(stats, indent=4))

# evaluate best POMDP agent with DQN initial positions
pomdp_env = POMDPDeformedGridworld(
    render_mode="rgb_array",
    obs_type='cardinal',
)

obs_model = load_obs_model('cardinal')
POMDPagent = TS(MDP_AGENT, pomdp_env, discretization=DISCRETIZATION, update=BELIEFUPDATE, obs_model=obs_model, debug=True) 


def eval_agent_pomdp_x0(agent,env: POMDPDeformedGridworld,num_episodes):
    """Returns
        - episode_transition: list of list of tuples (s,a,r,s',done), t[i] is the ith episode
        - beliefs: list of beliefs at each time step 
    """

    assert agent.debug, 'Agent must be in debug mode to evaluate'

    transitions = []
    info_steps_pomdp = []
    info_steps_mdp = []
    entropy = []

    for i in trange(num_episodes):

        agent.reset()
        s, _ = env.reset()

        starting_pos = mdp_transitions[i][0][0]['pos']
        deformation = mdp_transitions[i][0][0]['theta']
        env.set_deformation([deformation[0], deformation[3]],[deformation[1],deformation[2]])
        env.set_position(starting_pos)
        s = env.get_state()

        assert torch.allclose(s['pos'], torch.tensor(starting_pos)), f'Invalid starting position, got {s["pos"]} expected {starting_pos}'

        totalReward = 0.0
        done = False
        steps = 0
        episode_transitions = []
        episode_entropy = []
        
        while not done:

            best_action, _ = agent.predict(s, deterministic=True)

            next_state, reward, terminated, truncated, info = env.step(best_action)
            
            done = terminated or truncated
            s = next_state

            steps += 1
            totalReward += reward
            episode_transitions.append((s, best_action, reward, next_state, terminated, truncated))
            # episode_entropy.append(agent.entropy.item())

        transitions.append(episode_transitions)
        info_steps_pomdp.append(steps)
        info_steps_mdp.append(len(mdp_transitions[i]))
        assert mdp_steps[i] == len(mdp_transitions[i]), f'Invalid number of steps, got {steps} expected {len(mdp_transitions[i])}'
        # print(f"Episode {i}: T_POMDP(x_0) / T_MDP(x_0) = {tpomdp_tmdp}")

        # entropy.append(episode_entropy)

    env.close()

    return transitions, info_steps_pomdp, info_steps_mdp


pomdp_transitions, info_steps_pomdp, info_steps_mdp = eval_agent_pomdp_x0(POMDPagent,pomdp_env,N_EPISODES)


stats = evaluate_statistics(pomdp_transitions)
# insert stats into dataframe
data = data._append({
    'agent': MDP_AGENT.__class__.__name__,
    'model': None,
    'obs_type': None,
    'update': None,
    'discretization': None,
    'mean_episode_reward': stats['mean_episode_reward'],
    'std_episode_reward': stats['std_episode_reward'],
    'mean_episode_steps': stats['mean_episode_steps'],
    'std_episode_steps': stats['std_episode_steps'],
    'num_episodes': stats['num_episodes'],
    'terminated_episodes': stats['terminated_episodes'],
    'truncated_episodes': stats['truncated_episodes'],
    'termination_rate': stats['termination_rate'],
    'truncation_rate': stats['truncation_rate']
}, ignore_index=True)

# save dataframe to file
data.to_csv("results.csv", index=False)

print(json.dumps(stats, indent=4))


# plot info_steps 
import matplotlib.pyplot as plt 
import numpy as np
#
plt.plot(info_steps_pomdp, label='T_POMDP' )
plt.legend()
plt.savefig("eval/info_steps_pomdp.png")
plt.close()

plt.plot(info_steps_mdp, label='T_MDP')
plt.legend()
plt.savefig("eval/info_steps_mdp.png")
plt.close()
frac = np.array(info_steps_mdp) / np.array(info_steps_pomdp)
plt.plot(frac, label=' T_MDP / T_POMDP')
plt.legend()
plt.savefig("eval/info_steps_ratio.png")
plt.close()

print(f"Mean ratio: {np.mean(frac)}")
print(f"Std ratio: {np.std(frac)}")

invfrac = np.array(info_steps_pomdp) / np.array(info_steps_mdp)
plt.plot(invfrac, label=' T_POMDP / T_MDP')
plt.legend()
plt.savefig("eval/info_steps_inv_ratio.png")
plt.close()
print(f"Mean ratio: {np.mean(invfrac)}")
print(f"Std ratio: {np.std(invfrac)}")



# find indexes of the episodes with the highest values in info_steps_pomdp
pomdp_idx = np.argsort(info_steps_pomdp)
mdp_idx = np.argsort(info_steps_mdp)

limit = stats['truncated_episodes'] if stats['truncated_episodes'] > 0 else 2
idx = pomdp_idx[-limit:].tolist() + mdp_idx[-limit:].tolist() 
print(idx)
# select from info_steps_pomdp every entry except the ones in idx
info_steps_pomdp = np.array(info_steps_pomdp)
info_steps_mdp = np.array(info_steps_mdp)

info_steps_pomdp = info_steps_pomdp[~np.isin(np.arange(len(info_steps_pomdp)), idx)]
info_steps_mdp = info_steps_mdp[~np.isin(np.arange(len(info_steps_mdp)), idx)]

plt.plot(info_steps_pomdp, label='T_POMDP')
plt.legend()
plt.savefig("eval/info_steps_pomdp_clean.png")
plt.close()

plt.plot(info_steps_mdp, label='T_MDP')
plt.legend()
plt.savefig("eval/info_steps_mdp_clean.png")
plt.close()

frac = np.array(info_steps_mdp) / np.array(info_steps_pomdp)
plt.plot(frac, label=' T_MDP / T_POMDP')
plt.legend()
plt.savefig("eval/info_steps_ratio_clean.png")
plt.close()
print(f"Mean ratio: {np.mean(frac)}")
print(f"Std ratio: {np.std(frac)}")

invfrac = np.array(info_steps_pomdp) / np.array(info_steps_mdp)
plt.plot(invfrac, label=' T_POMDP / T_MDP')
plt.legend()
plt.savefig("eval/info_steps_inv_ratio.png")
plt.close()

print(f"Mean ratio: {np.mean(invfrac)}")
print(f"Std ratio: {np.std(invfrac)}")

# plt.plot(np.array(info_steps_pomdp) / np.array(info_steps_mdp), label='T_POMDP(x_0) / T_MDP(x_0)')