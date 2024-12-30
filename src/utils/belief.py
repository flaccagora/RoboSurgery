import torch
import itertools
import numpy as np

obs_dict = {obs:i for i, obs in enumerate(list(itertools.product([0,1], repeat=5)))}


def belief_entropy(probabilities):
    
    # Calculate entropy, avoiding log(0) by adding a mask
    entropy = -torch.sum(probabilities * torch.log2(probabilities + (probabilities == 0).float()))
    
    return entropy.item()  # .item() to get a standard Python float

def b_theta_update(env, belief,pos,observation):
    """
    perform update over theta
    
    $$b'_{x,a,o}(theta) = \eta \cdot p(o|x,theta) \cdot b(theta)$$
    
    """

    new_belief = torch.zeros_like(belief)

    for t, theta in enumerate(env.deformations):
        P_o_s_theta = np.all(env.get_observation((pos,theta)) == observation) # 0 or 1 

        new_belief[t] = P_o_s_theta * belief[t]
    
    new_belief = new_belief / (torch.sum(new_belief) + 1e-10)

    return new_belief


def update_belief(belief, action, observation, T, O):
    """
    Perform a Bayesian belief update in a POMDP with action-dependent transition and observation models.

    Parameters:
        belief (torch.Tensor): Initial belief distribution over states, shape (num_states,)
        action (int): The action taken (index of action)
        observation (int): The observation received (index of observation)
        T (torch.Tensor): Transition probabilities, shape (num_states, num_actions, num_states)
        O (torch.Tensor): Observation probabilities, shape (num_states, num_actions, num_observations)

    Returns:
        torch.Tensor: The updated belief over states, shape (num_states,)
    """
    # Prediction Step: Compute predicted belief over next states
    predicted_belief = torch.matmul(belief, T[:, action])

    # Update Step: Multiply by observation likelihood
    # observation_likelihood = O[:, action, obs_dict[tuple(observation.tolist())]]
    observation_likelihood = O[:, action, observation]

    new_belief = predicted_belief * observation_likelihood

    # Normalize the updated belief to ensure it's a valid probability distribution
    if new_belief.sum() > 0:
        new_belief /= new_belief.sum() 
            
    return new_belief


import torch
import torch.nn as nn
import torch.distributions as dist
from torch.optim import Adam
import numpy as np

class BetaVariationalBayesianInference:
    def __init__(self, f, input_dim, latent_dim=1, hidden_dim=32):
        """
        Initialize the variational Bayesian inference model with Beta distributions.
        
        Args:
            f: callable, the known function linking X and y through theta
            input_dim: int, dimension of input X
            latent_dim: int, dimension of latent parameter theta
            hidden_dim: int, dimension of hidden layers in the neural network
        """
        self.f = f
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        # Variational parameters (alpha and beta of q(theta))
        # Initialize to reasonable values (alpha=beta=2 gives a symmetric Beta)
        self.q_alpha = nn.Parameter(2 * torch.ones(latent_dim))
        self.q_beta = nn.Parameter(2 * torch.ones(latent_dim))
        
        # Prior parameters (can be customized)
        self.prior_alpha = torch.ones(latent_dim)  # Default to Beta(1,1) = Uniform(0,1)
        self.prior_beta = torch.ones(latent_dim)
        
        self.low = torch.tensor([.4, -.2, -.2, .4])
        self.high = torch.tensor([1.0, .2, .2, 1.0])

    def sample_latent(self, n_samples=1):
        """
        Sample from the variational distribution q(theta) using the Beta distribution
        """
        q_dist = dist.Beta(self.q_alpha, self.q_beta)
        theta =  q_dist.rsample((n_samples,))
        return self.low + (self.high - self.low) * theta
    
    def elbo(self, X, y, n_samples=10):
        """
        Compute the evidence lower bound (ELBO) with Beta distributions
        
        Args:
            X: torch.Tensor, input data (batch_size, input_dim)
            y: torch.Tensor, observations (batch_size,)
            n_samples: int, number of Monte Carlo samples
        """
        batch_size = X.shape[0]
        
        # Sample from variational distribution
        theta_samples = self.sample_latent(n_samples)  # (n_samples, latent_dim)
        
        # Compute log likelihood for each sample
        log_likelihood = torch.zeros(n_samples, batch_size)
        for i in range(n_samples):
            theta = theta_samples[i]
            y_pred = self.f(X, theta.expand(batch_size, -1)).squeeze()
            log_likelihood[i] = dist.Bernoulli(y_pred).log_prob(y)
        
        # Average over samples
        expected_log_likelihood = torch.mean(log_likelihood, dim=0).sum()
        
        # Compute KL divergence between Beta distributions
        q_dist = dist.Beta(self.q_alpha, self.q_beta)
        prior_dist = dist.Beta(self.prior_alpha, self.prior_beta)
        kl_div = dist.kl_divergence(q_dist, prior_dist).sum()
        
        return expected_log_likelihood - kl_div
    
    def fit(self, X, y, n_epochs=100, batch_size=64, lr=0.1):
        """
        Fit the model using variational inference
        
        Args:
            X: torch.Tensor, input data
            y: torch.Tensor, observations
            n_epochs: int, number of training epochs
            batch_size: int, batch size for stochastic optimization
            lr: float, learning rate
        """
        optimizer = Adam([self.q_alpha, self.q_beta], lr=lr)
        
        dataset = torch.utils.data.TensorDataset(X, y)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        for epoch in range(n_epochs):
            epoch_loss = 0
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                loss = -self.elbo(batch_X, batch_y, n_samples=100)  # Negative because we minimize
                loss.backward()
                optimizer.step()
                
                # Ensure parameters stay positive
                with torch.no_grad():
                    self.q_alpha.data.clamp_(min=1e-6)
                    self.q_beta.data.clamp_(min=1e-6)
                
                epoch_loss += loss.item()
            
            if (epoch + 1) % 10 == 0:
                mean = self.q_alpha.detach() / (self.q_alpha.detach() + self.q_beta.detach())
                mean = self.low + (self.high - self.low) * mean
                print(f"Epoch {epoch+1}/{n_epochs}, Loss: {epoch_loss/len(dataloader):.4f}")
                print(f"Estimated theta mean: {mean}")
    
    def get_posterior_params(self):
        """Return the learned posterior parameters"""
        return {
            'alpha': self.q_alpha.detach(),
            'beta': self.q_beta.detach(),
            'mean': self.low + (self.high - self.low) * (self.q_alpha / (self.q_alpha + self.q_beta)).detach(),
            'mode': ((self.q_alpha - 1) / (self.q_alpha + self.q_beta - 2)).detach()
        }