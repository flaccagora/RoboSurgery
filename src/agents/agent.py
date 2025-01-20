import torch
import numpy as np
from collections import OrderedDict
from stable_baselines3 import PPO
from environment.env import POMDPDeformedGridworld

class POMDPAgent():
    
    def __init__(self,mdp_agent:PPO, pomdp_env: POMDPDeformedGridworld, discretization=10, update='discrete', obs_model=None, debug=False):
        """
            mdp_agent: pretrained MDP agent to use for the POMDP agent
            pomdp_env: POMDP environment
            discretization: number of points to discretize the belief space if using discrete belief update
                            n_particles if using particle filters
            update: belief update method, one of ['discrete', 'discrete_exact', 'variational', 'particlefilters']
            obs_model: observation model for the POMDP environment gives back p such thath Bernoulli(p).log_prob(obs) gives the likelihood of the observation
            debug: print debug information
        
        """
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
            
            self.original_def = pomdp_env.transformation_matrix[0][0], pomdp_env.transformation_matrix[1][1]

        if update == 'discrete': 
            assert obs_model is not None, f'Need an observation model for discrete belief update, given {obs_model}'
            self.obs_model = obs_model
            self.belief_update = self.discrete_belief_update
            print('Discrete belief update with observation model - SHEAR allowed')
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
            self.PF.initialize_particles()
            self.belief_update = self.particle_filter_belief_update
            
            self.X_history = [self.pomdp_env.get_state()['pos']]
            self.y_history = [self.pomdp_env.get_state()['obs']]

        else:
            raise ValueError('Invalid belief update method')
        
        self.debug = debug
        if self.debug:
            print(f'Using {update} belief update method')
    
    def predict(self, s, deterministic=True):
        
        self.belief_update(s)
        pos = s['pos']
        if self.update == 'discrete_exact' or self.update == 'discrete':
            # theta = self.belief_points[self.belief_values.argmax()] # QMDP
            theta = self.belief_points[torch.multinomial(self.belief_values, 1).item()] # Thompson sampling
        elif self.update == 'variational':
            theta = self.VI.sample_latent(1).squeeze().clone().detach().numpy() # variational Thompson sampling
        elif self.update == 'particlefilters':
            mean, var = self.PF.estimate_posterior()
            theta = torch.distributions.Normal(torch.tensor(mean), torch.tensor(var).sqrt()+1e-6).sample().squeeze()
            # theta = torch.tensor(mean, dtype=torch.float32)

        s = OrderedDict({'pos': pos, 'theta': theta})
        action = self.mdp_agent.predict(s, deterministic=deterministic)

        self.on_precidt_callback()
        return action
    
    def discrete_belief_update(self, pomdp_state):
        """discrete belief update"""
        pos = pomdp_state['pos']
        obs = pomdp_state['obs']

        batch_pos = pos.repeat(len(self.belief_points), 1)

        with torch.no_grad():
            predictions = self.obs_model(batch_pos,self.belief_points)

        likelihood = torch.exp(torch.distributions.Bernoulli(predictions).log_prob(obs))
        if len(likelihood.shape) == 2:
            likelihood = likelihood.sum(dim=1)

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

    def particle_filter_belief_update(self, pomdp_state):
        self.X_history.append(pomdp_state['pos'])
        self.y_history.append(pomdp_state['obs'])

        # X = posizione dell'agente (x,y)
        X = torch.stack(self.X_history[-1:])

        # ossevrazioni dell'agente negli stati pos=(x,y)
        y = torch.stack(self.y_history[-1:])

        # X, y = pomdp_state['pos'].unsqueeze(0), pomdp_state['obs'].unsqueeze(0)

        # Create and fit the model
        self.PF.update(X, y)

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
            self.entropy = None# self.PF.entropy()
        
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
            self.PF.initialize_particles()