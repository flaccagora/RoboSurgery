import numpy as np
import gymnasium as gym
import torch

import os 
import sys
from tqdm import trange
import json

N_EPISODES = 1000 # number of episodes to average over
OBSERVATION_TYPES = ['cardinal', 'single'] # 'cardinal' or  'single'
BELIEF_UPDATES = ['particlefilters'] # variational escluso per ora
# BELIEF_UPDATES = ['discrete', 'particlefilters'] # variational escluso per ora

# DISCRETIZATION = {'discrete': [5], 'variational': 10, 'particlefilters': [1000]} 
DISCRETIZATION = {'discrete': [5,10], 'variational': 10, 'particlefilters': [1000,2000,5000,10000]} 
DEBUG = True

# Load MDP solution -------------------------------------------------------------------------

run = "PPO_continous_" + "enh53x0u"

from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3 import PPO, DQN
from environment.env import Grid
from utils.checkpoints import find_last_checkpoint
from collections import OrderedDict
from environment.env import POMDPDeformedGridworld

env = Grid(
    shear_range=(-.2, .2),
    stretch_range=(.4,1),
    render_mode="rgb_array"
)

last_checkpoint = find_last_checkpoint(f"agents/pretrained/MDP/{run}")
PPO_model = PPO.load(f"agents/pretrained/MDP/{run}/{last_checkpoint}", env=env)

run = "DQN_continous_" + "e2qthdat"

last_checkpoint = find_last_checkpoint(f"agents/pretrained/MDP/{run}")
DQN_model = DQN.load(f"agents/pretrained/MDP/{run}/{last_checkpoint}", env=env)

env.close()
MODELS = [PPO_model, DQN_model]
MODELS = [DQN_model]

# ------------------------------------------------------------------------------------------------

class BaseAgent():
    
    def __init__(self,mdp_agent:PPO, pomdp_env: POMDPDeformedGridworld, discretization=10, update='discrete', obs_model=None, debug=False):
        assert isinstance(pomdp_env, POMDPDeformedGridworld), f'Invalid environment type {type(pomdp_env)}'
        self.pomdp_env = pomdp_env
        self.mdp_agent = mdp_agent
        self.update = update

        if update == 'discrete' or update == 'discrete_exact':
            stretch = np.linspace(.4, 1, discretization)
            shear = np.linspace(-.2,.2, discretization)
            xa,ya,yb,xb = np.meshgrid(stretch, shear,shear,stretch) # , shear, shear
            positions = np.column_stack([xa.ravel(),ya.ravel(),yb.ravel(),xb.ravel()]),
            positions = torch.tensor(np.array(positions), dtype=torch.float32)
            self.belief_points = positions.squeeze()
            self.belief_values = torch.ones(self.belief_points.shape[0], dtype=torch.float32, requires_grad=False) / len(positions)
            
            self.original_def = env.transformation_matrix[0][0], env.transformation_matrix[1][1]

        if update == 'discrete': 
            assert obs_model is not None, f'Need an observation model for discrete belief update, given {obs_model}'
            self.obs_model = obs_model
            self.belief_update = self.discrete_belief_update
            # print('Discrete belief update with observation model - SHEAR allowed')
        elif update == 'discrete_exact':
            self.belief_update = self.exact_belief_update
            raise NotImplementedError('Exact belief update not implemented here')
        elif update == 'variational':
            from utils.belief import BetaVariationalBayesianInference
            assert obs_model is not None, f'Need an observation model for variational belief update, given {obs_model}'
            self.VI = BetaVariationalBayesianInference(obs_model, input_dim=2, latent_dim=4, debug=debug)
            self.obs_model = obs_model

            self.belief_update = self.variational_belief_update
            self.X_history = [self.pomdp_env.get_state()['pos']]
            self.y_history = [self.pomdp_env.get_state()['obs']]
        elif update == 'particlefilters':
            from utils.belief import BayesianParticleFilter
            self.obs_model = obs_model
            self.n_particles = discretization
            self.PF = BayesianParticleFilter(f = obs_model, n_particles=self.n_particles, theta_dim=4)
            self.belief_points, self.belief_values = self.PF.initialize_particles()
            self.belief_update = self.particle_filter_belief_update
            
            self.X_history = [self.pomdp_env.get_state()['pos']]
            self.y_history = [self.pomdp_env.get_state()['obs']]

        else:
            raise ValueError('Invalid belief update method')
        
        self.debug = debug
        # if self.debug:
            # print(f'Using {update} belief update method')
    
    def predict(self, s, deterministic=True):
        raise NotImplementedError('Predict method not implemented')
    
    def discrete_belief_update(self, pomdp_state):
        """discrete belief update"""
        pos = pomdp_state['pos']
        obs = pomdp_state['obs']

        batch_pos = pos.repeat(len(self.belief_points), 1)

        model_device = next(self.obs_model.parameters()).device
        with torch.no_grad():
            predictions = self.obs_model(batch_pos.to(model_device),self.belief_points.to(model_device))

        likelihood = torch.distributions.Bernoulli(predictions).log_prob(obs.to(model_device))
        if len(likelihood.shape) == 2:
            likelihood = likelihood.sum(dim=1)
        likelihood = likelihood.exp()

        tmp = likelihood.squeeze() * self.belief_values.to(model_device)
        self.belief_values = tmp  / tmp.sum()

    def exact_belief_update(self, pomdp_state):
        """discrete belief update"""
        obs = pomdp_state['obs']
        pos = pomdp_state['pos']

        def f():
            likelihood = []
            for x in self.belief_points:
                try:
                    self.pomdp_env.set_deformation([x[0], x[1]],[0,0]) # stretch, shear format
                    likelihood.append(torch.all(torch.tensor(self.pomdp_env.observe(list(pos))) == obs))
                except:
                    raise ValueError('Invalid belief point x', x)
            self.pomdp_env.set_deformation(self.original_def, [0,0])
            return torch.tensor(likelihood, dtype=torch.float32)

        
        likelihood = f()
        self.belief_values =  likelihood * self.belief_values
        self.belief_values = self.belief_values / self.belief_values.sum()

    def variational_belief_update(self, pomdp_state):
        self.X_history.append(pomdp_state['pos'])
        self.y_history.append(pomdp_state['obs'])

        # X = posizione dell'agente (x,y)
        X = torch.stack(self.X_history)

        # ossevrazioni dell'agente negli stati pos=(x,y)
        y = torch.stack(self.y_history)

        # Create and fit the model
        self.VI.fit(X, y, n_epochs=10, lr=0.05)

    def particle_filter_belief_update(self, pomdp_state):
        self.X_history.append(pomdp_state['pos'])
        self.y_history.append(pomdp_state['obs'])

        # X = posizione dell'agente (x,y)
        X = torch.stack(self.X_history[-1:])

        # ossevrazioni dell'agente negli stati pos=(x,y)
        y = torch.stack(self.y_history[-1:])

        # X, y = pomdp_state['pos'].unsqueeze(0), pomdp_state['obs'].unsqueeze(0)

        # Create and fit the model
        self.belief_points, self.belief_values = self.PF.update(X, y)

    def on_precidt_callback(self):
        if self.debug:
            self.print_stats()
        
    def print_stats(self):
        if self.update == 'discrete':
            # print(f'Belief shape: {self.belief_values.shape}')
            # print(f'Belief points shape: {self.belief_points.shape}')
            # print(f'Belief max: {self.belief_points[self.belief_values.argmax()]}')
            # print(f'Belief sum: {self.belief_values.sum()}')
            # print(f'Belief entropy: {torch.distributions.Categorical(probs=self.belief_values).entropy()}')
            # print("\n")
            self.entropy = torch.distributions.Categorical(probs=self.belief_values).entropy()
        elif self.update == 'variational':
            # print(f'Variational inference: {self.VI.entropy()}')
            # print(self.VI.get_posterior_params())
            # print("\n")
            self.entropy = self.VI.entropy()
        elif self.update == 'particlefilters':
            # print(f'Particle filter: {self.PF.estimate_posterior()[1]}')
            # print("\n")
            self.entropy = self.PF.entropy()
        
    def reset(self):
        self.X_history = [self.pomdp_env.get_state()['pos']]
        self.y_history = [self.pomdp_env.get_state()['obs']]
        self.entropy = None

        if self.update == 'discrete':
            self.belief_values = torch.ones(self.belief_points.shape[0], dtype=torch.float32, requires_grad=False) / len(self.belief_points)
        elif self.update == 'variational':
            del self.VI
            from utils.belief import BetaVariationalBayesianInference
            self.VI = BetaVariationalBayesianInference(self.obs_model, input_dim=2, latent_dim=4, debug=self.debug)
        elif self.update == 'particlefilters':
            del self.PF
            from utils.belief import BayesianParticleFilter
            self.PF = BayesianParticleFilter(f = self.obs_model, n_particles=self.n_particles, theta_dim=4)
            self.belief_points, self.belief_values = self.PF.initialize_particles()

class TS(BaseAgent):
    def __init__(self, mdp_agent:PPO, pomdp_env: POMDPDeformedGridworld, discretization=10,
                 update='discrete', obs_model=None, debug=False):
        super().__init__(mdp_agent, pomdp_env, discretization, update, obs_model, debug)


    def predict(self, s, deterministic=True):
        
        self.belief_update(s)
        pos = s['pos']
        if self.update == 'discrete_exact' or self.update == 'discrete':
            theta = self.belief_points[torch.multinomial(self.belief_values, 1).item()] #  sampling
        elif self.update == 'variational':
            theta = self.VI.sample_latent(1).squeeze().clone().detach().numpy() # variational sampling
        elif self.update == 'particlefilters':
            mean, var = self.PF.estimate_posterior()
            theta = torch.distributions.Normal(torch.tensor(mean), torch.tensor(var).sqrt()+1e-6).sample().squeeze()

        s = OrderedDict({'pos': pos, 'theta': theta})
        action = self.mdp_agent.predict(s, deterministic=deterministic)

        self.on_precidt_callback()
        return action

class MLS(BaseAgent):
    
    def __init__(self, mdp_agent:PPO, pomdp_env: POMDPDeformedGridworld, discretization=10,
                update='discrete', obs_model=None, debug=False):
        super().__init__(mdp_agent, pomdp_env, discretization, update, obs_model, debug)

    def predict(self, s, deterministic=True):
        
        self.belief_update(s)
        
        if self.update == 'discrete_exact' or self.update == 'discrete':
            theta = self.belief_points[torch.argmax(self.belief_values).item()] 
        elif self.update == 'variational':
            theta = self.VI.sample_latent(1).squeeze().clone().detach().numpy() # variational sampling
        elif self.update == 'particlefilters':
            mean, var = self.PF.estimate_posterior()
            theta = torch.tensor(mean, dtype=torch.float32)

        pos = s['pos']
        s = OrderedDict({'pos': pos, 'theta': theta})
        action = self.mdp_agent.predict(s, deterministic=deterministic)

        self.on_precidt_callback()
        return action
    
class QMDP(BaseAgent):
    def __init__(self,mdp_agent:DQN, pomdp_env: POMDPDeformedGridworld, discretization=10, update='discrete', obs_model=None, debug=False):
        super().__init__(mdp_agent, pomdp_env, discretization, update, obs_model, debug)

        self.Q = self.mdp_agent.policy.q_net
        self.obs_model.to(self.mdp_agent.device)

        # if update == 'particlefilters':
        #     self.belief_points = self.PF.particles
        #     self.belief_values = self.PF.weights
    
    def predict(self, s, deterministic=True):
        """
            QMDP works as follows:
            1. Compute Q(s,a) for all belief points and actions
            2. Compute QMDP(s) = argmax_a \sum_{theta} b(theta) * Q(s,a)
            3. Return QMDP(s)
        """
        self.belief_update(s)
        
        # assert torch.allclose(self.belief_values.to(self.mdp_agent.device), self.PF.weights.to(self.mdp_agent.device)), 'Invalid belief values'
        
        # Move belief points to device if needed
        belief_points = self.belief_points.to(self.mdp_agent.device)
        belief_values = self.belief_values.to(self.mdp_agent.device)
        
        # Step 1: Compute Q(s,a) for all belief points and actions
        B = belief_values.shape[0] # number of belief points Batch
        state = OrderedDict({'pos': s['pos'].expand(B,-1).to(self.mdp_agent.device), 'theta': belief_points}) 
        # print(state['pos'].shape, state['theta'].shape)
        # print(self.Q(state).shape)
        # print(self.belief_values.shape)
        qmdp = self.Q(state)
        # Step 2: Compute QMDP(s) = argmax_a \sum_{theta} b(theta) * Q(s,a)

        actions = torch.einsum("s,sa->a",belief_values, qmdp)
        # print(qmdp.shape)

        if deterministic:
            return torch.argmax(actions).item(), actions
        else:
            # Implement stochastic policy if needed
            probs = torch.softmax(actions / 0.1, dim=0)  # Temperature parameter set to 0.1
            return torch.multinomial(probs, 1).item(), actions

AGENTS = [TS, MLS, QMDP]
AGENTS = [QMDP]


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

def eval_agent_pomdp(agent:BaseAgent,env: POMDPDeformedGridworld,num_episodes):
    """Returns
        - episode_transition: list of list of tuples (s,a,r,s',done), t[i] is the ith episode
        - beliefs: list of beliefs at each time step 
    """

    assert agent.debug, 'Agent must be in debug mode to evaluate'

    transitions = []
    entropy = []

    for i in trange(num_episodes):

        agent.reset()
        s, _ = env.reset()

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
        # entropy.append(episode_entropy)

    env.close()

    return transitions

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

import pandas as pd
data = pd.DataFrame(columns=['agent', 'model', 'obs_type', 'update', 'discretization', 'mean_episode_reward', 'std_episode_reward', 'mean_episode_steps', 'std_episode_steps', 'num_episodes', 'terminated_episodes', 'truncated_episodes', 'termination_rate', 'truncation_rate'])

for obs_type in OBSERVATION_TYPES:
    obs_model = load_obs_model(obs_type)
    env = POMDPDeformedGridworld(
            render_mode="rgb_array",
            obs_type=obs_type
        )
    for update in BELIEF_UPDATES:
        for MDPmodel in MODELS:
            for agent in AGENTS:
                # Skip QMDP with PPO as it's designed to work with DQN
                if agent == QMDP and isinstance(MDPmodel, PPO):
                    print(f"Skipping QMDP with PPO model as QMDP requires a DQN model")
                    continue

                
                for discretization in DISCRETIZATION[update]:
                    
                    solver = agent(MDPmodel, env, discretization=
                                discretization, update=update, obs_model=obs_model, debug=DEBUG)
                    
                    message = ["-"*50, f"Running {solver.__class__.__name__} \n MDP = {MDPmodel} \n obstype = {obs_type} \n update = {update} \n discretization = {discretization}"]
                    
                    print("\n".join(message))
                                        
                    transitions = eval_agent_pomdp(solver, env, num_episodes=N_EPISODES)
                    stats = evaluate_statistics(transitions)
                    # insert stats into dataframe
                    data = data._append({
                        'agent': solver.__class__.__name__,
                        'model': MDPmodel.__class__.__name__,
                        'obs_type': obs_type,
                        'update': update,
                        'discretization': discretization,
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

                    