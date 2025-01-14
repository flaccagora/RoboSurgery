import torch
import numpy as np
from collections import OrderedDict
from stable_baselines3 import PPO

from environment.env import POMDPDeformedGridworld

class POMDPAgent(PPO):
    
    def __init__(self, pomdp_env: POMDPDeformedGridworld, discretization=10, update='discrete_exact', obs_model=None):
        super().__init__('MlpPolicy')
        assert isinstance(pomdp_env, POMDPDeformedGridworld), f'Invalid environment type {type(env)}'
        self.pomdp_env = pomdp_env

        if update == 'discrete_modelled' or update == 'discrete_exact':
            stretch = np.linspace(.4, 1, discretization)
            # shear = np.linspace(0,0, discretization)
            xa,xb = np.meshgrid(stretch, stretch) # , shear, shear
            positions = np.column_stack([xa.ravel(),xb.ravel()]), #  ya.ravel(),yb.ravel()
            positions = torch.tensor(positions, dtype=torch.float32)
            self.belief_points = positions.squeeze()
            self.belief_values = torch.ones(self.belief_points.shape[0], dtype=torch.float32) / len(positions)

        if update == 'discrete_modelled': 
            assert obs_model is not None, f'Need an observation model for discrete_modelled belief update, given {obs_model}'
            self.obs_model = obs_model
            self.belief_update = self.discrete_belief_update
            print('Discrete belief update with observation model - no shear allowed')
        elif update == 'discrete_exact':
            self.belief_update = self.exact_belief_update
            print('Discrete belief update without observation model - no shear allowed')
        elif update == 'variational':
            from utils.belief import BetaVariationalBayesianInference
            assert obs_model is not None, f'Need an observation model for variational belief update, given {obs_model}'
            self.VI = BetaVariationalBayesianInference(obs_model, input_dim=2, latent_dim=4)

            self.belief_update = self.variational_belief_update
            self.X_history = [self.pomdp_env.get_state()['pos']]
            self.y_history = [self.pomdp_env.get_state()['obs']]
        else:
            raise ValueError('Invalid belief update method')
        
        self.original_def = env.transformation_matrix[0][0], env.transformation_matrix[1][1]
    
    def predict(self, s, deterministic=True):
        self.belief_update(self.pomdp_env.get_state())
        
        pos = self.pomdp_env.get_state()['pos']
        theta = self.belief_points[self.belief_values.argmax()] 
        s = OrderedDict({'pos': pos, 'theta': theta})
        
        action = super().predict(s, deterministic=deterministic)
        return action

    def discrete_belief_update(self, pomdp_state):
        """discrete belief update"""
        pos = pomdp_state['pos']
        obs = pomdp_state['obs']

        batch_pos = pos.repeat(len(self.belief_points), 1)
        
        # need theta because working on two parameters only in this example
        theta = torch.cat([self.belief_points, torch.zeros(len(self.belief_points), 2)], dim=1)
        # permute theta to match the order of pos
        theta = theta[:, [0,3,2,1]]
        

        predictions = self.obs_model(batch_pos,theta)
        likelihood = torch.exp(torch.distributions.Bernoulli(predictions).log_prob(obs))

        tmp = likelihood.squeeze() * self.belief_values
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

