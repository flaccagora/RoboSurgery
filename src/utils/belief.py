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
    def __init__(self, f, input_dim=2, latent_dim=4, debug=False):
        """
        Initialize the variational Bayesian inference model with Beta distributions.
        
        Args:
            f: callable, the known function linking X and y through theta
            input_dim: int, dimension of input X
            latent_dim: int, dimension of latent parameter theta
        """
        self.f = f
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.debug = debug
        
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
        # log_likelihood = torch.zeros(n_samples, batch_size)
        # for i in range(n_samples):
        #     theta = theta_samples[i]
        #     y_pred = self.f(X, theta.expand(batch_size, -1)).squeeze()
        #     if len(y_pred.shape) == 1:
        #         log_likelihood[i] = dist.Bernoulli(y_pred).log_prob(y)
        #     elif len(y_pred.shape) == 2:
        #         log_likelihood[i] = dist.Bernoulli(y_pred).log_prob(y).sum(dim=1)
        # # Average over samples
        # expected_log_likelihood = torch.mean(log_likelihood, dim=0).sum()
        
        theta_expanded = theta_samples.unsqueeze(1).expand(-1, batch_size, -1)  # Expand theta for batch computation
        X_expanded = X.unsqueeze(0).expand(n_samples, -1, -1)  # Expand X for batch computation
        # print(X_expanded.shape, theta_expanded.shape)
        y_pred = self.f(X_expanded, theta_expanded,dim=2)  # Shape: (n_particles, batch_size)
        y_pred = y_pred.squeeze()  # Adjust shape if needed
        # print(y_pred.shape, y.shape)
        log_likelihoods = dist.Bernoulli(y_pred.view(-1)).log_prob(y.expand(n_samples,-1).reshape(-1)).reshape(n_samples,batch_size).sum(dim=-1)  # Sum over batch size
        expected_log_likelihood0 = log_likelihoods.mean(dim=0).sum()

        # Compute KL divergence between Beta distributions
        q_dist = dist.Beta(self.q_alpha, self.q_beta)
        prior_dist = dist.Beta(self.prior_alpha, self.prior_beta)
        kl_div = dist.kl_divergence(q_dist, prior_dist).sum()
        
        return expected_log_likelihood0 - kl_div
    
    def fast_elbo(self, X, y, n_samples=10):
        """
        Compute the evidence lower bound (ELBO) with Beta distributions.

        Args:
            X: torch.Tensor, input data (batch_size, input_dim)
            y: torch.Tensor, observations (batch_size,)
            n_samples: int, number of Monte Carlo samples
        """
        batch_size = X.shape[0]

        # Sample from variational distribution
        theta_samples = self.sample_latent(n_samples).repeat_interleave(batch_size, dim=0)  # (n_samples, latent_dim)

        # Expand input for vectorized computation
        X_expanded = X.expand(n_samples, -1, -1)  # (n_samples, batch_size, input_dim)

        print("X_expanded: ", X_expanded[:3])
        print("X_expanded shape: ", X_expanded.shape)
        print("theta_samples: ", theta_samples[:3])
        print("theta_samples shape: ", theta_samples.shape)
        # Compute predictions for all samples at once
        y_preds = self.f(X_expanded, theta_samples)  # (n_samples, batch_size, output_dim)



        # Compute log-likelihood in a vectorized manner
        if len(y_preds.shape) == 2:  # Binary classification case
            log_likelihood = dist.Bernoulli(y_preds).log_prob(y).sum(dim=-1)  # (n_samples, batch_size)
        elif len(y_preds.shape) == 3:  # Multi-output case
            log_likelihood = dist.Bernoulli(y_preds).log_prob(y.unsqueeze(-1)).sum(dim=-1).sum(dim=-1)  # (n_samples, batch_size)

        # Average over samples and sum over batch
        expected_log_likelihood = log_likelihood.mean(dim=0).sum()

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
                # fast_loss = -self.fast_elbo(batch_X, batch_y, n_samples=100)
                # assert torch.isclose(loss, fast_loss), f"ELBO values do not match: {loss} vs {fast_loss}"
                loss.backward()

                optimizer.step()
                
                # Ensure parameters stay positive
                with torch.no_grad():
                    self.q_alpha.data.clamp_(min=1e-6)
                    self.q_beta.data.clamp_(min=1e-6)
                
                epoch_loss += loss.item()
            
            # if self.debug and (epoch + 1) % 10 == 0:
            #     mean = self.q_alpha.detach() / (self.q_alpha.detach() + self.q_beta.detach())
            #     mean = self.low + (self.high - self.low) * mean
            #     print(f"Epoch {epoch+1}/{n_epochs}, Loss: {epoch_loss/len(dataloader):.4f}")
            #     print(f"Estimated theta mean: {mean}")
    
    def get_posterior_params(self):
        """Return the learned posterior parameters"""
        return {
            'alpha': self.q_alpha.detach(),
            'beta': self.q_beta.detach(),
            'mean': self.low + (self.high - self.low) * (self.q_alpha / (self.q_alpha + self.q_beta)).detach(),
            'mode': ((self.q_alpha - 1) / (self.q_alpha + self.q_beta - 2)).detach()
        }
    
    def entropy(self):

        """Compute the entropy of the variational distribution"""
        q_dist = dist.Beta(self.q_alpha, self.q_beta)
        return q_dist.entropy().sum()
    
import torch
import torch.distributions as dist
import numpy as np

class BayesianParticleFilter:
    def __init__(self, f, n_particles=1000, theta_dim=4):
        """
        Initialize the particle filter for Bayesian linear regression with 2D parameters.
        
        Args:
            n_particles (int): Number of particles to use for approximating the posterior.
            theta_dim (int): Dimensionality of the parameter vector (default is 2).
        """
        self.f = f
        self.n_particles = n_particles
        self.theta_dim = theta_dim
        self.particles = None
        self.weights = None

        self.low = torch.tensor([.4, -.2, -.2, .4])
        self.high = torch.tensor([1.0, .2, .2, 1.0])

        
    def initialize_particles(self, prior_mean=0.0, prior_std=1.0):
        """
        Initialize particles from the prior distribution.
        
        Args:
            prior_mean (float): Mean of the prior distribution.
            prior_std (float): Standard deviation of the prior distribution.
        """
        self.q_alpha = torch.nn.Parameter(torch.ones(4))
        self.q_beta = torch.nn.Parameter(torch.ones(4))

        beta = dist.Beta(self.q_alpha, self.q_beta)
        self.particles = self.low + (self.high - self.low) * beta.sample((self.n_particles,))

        # Initialize uniform weights
        self.weights = torch.ones(self.n_particles) / self.n_particles
        
    def log_likelihood(self, X, y, theta, noise_std=0.1):
        """
        Compute the log likelihood log p(y|X,theta) assuming Gaussian noise.
        
        Args:
            X (torch.Tensor): Input features (n_samples, 2).
            y (torch.Tensor): Target values (n_samples,).
            theta (torch.Tensor): Parameter particles (n_particles, theta_dim).
            noise_std (float): Standard deviation of observation noise.
        
        Returns:
            torch.Tensor: Log likelihood values for each particle.
        """
        batch_size = X.shape[0]
        
        # # Compute log likelihood for each sample
        # log_likelihoods0 = torch.zeros(self.n_particles)
        # for i in range(self.n_particles):
        #     t = theta[i]
        #     y_pred = self.f(X, t.expand(batch_size, -1)).squeeze()
        #     log_likelihoods0[i] = dist.Bernoulli(y_pred).log_prob(y).sum()
        

        ########### test
        # Compute log likelihood for each sample in parallel
        theta_expanded = theta.unsqueeze(1).expand(-1, batch_size, -1)  # Expand theta for batch computation
        X_expanded = X.unsqueeze(0).expand(self.n_particles, -1, -1)  # Expand X for batch computation
        y_pred = self.f(X_expanded, theta_expanded,dim=2)  # Shape: (n_particles, batch_size)
        y_pred = y_pred.squeeze()  # Adjust shape if needed
        if y.shape[-1] == 4:
            log_likelihoods = dist.Bernoulli(y_pred.view(-1)).log_prob(y.expand(self.n_particles,-1,-1).reshape(-1)).reshape(self.n_particles,batch_size,4).sum(dim=-1).sum(dim=-1)  # Sum over batch size
        else:
            log_likelihoods = dist.Bernoulli(y_pred.view(-1)).log_prob(y.expand(self.n_particles,-1).reshape(-1)).reshape(self.n_particles,batch_size).sum(dim=-1)  # Sum over batch size
        ###########
        # Average over samples
        # assert torch.allclose(log_likelihoods0, log_likelihoods)
        return log_likelihoods
        
    def update(self, X, y, noise_std=0.1):
        """
        Update particle weights based on new observations.
        
        Args:
            X (torch.Tensor): Input features (n_samples, theta_dim).
            y (torch.Tensor): Target values (n_samples,).
            noise_std (float): Standard deviation of observation noise.
        """
        # Compute log likelihood for each particle
        log_likelihoods = self.log_likelihood(X, y, self.particles, noise_std)

        # Update log weights
        log_weights = torch.log(self.weights) + log_likelihoods
        
        # Subtract maximum for numerical stability before exp
        log_weights_normalized = log_weights - torch.max(log_weights)
        self.weights = torch.exp(log_weights_normalized)
        
        # Normalize weights
        self.weights /= self.weights.sum()
        
        # Compute effective sample size
        eff_sample_size = 1.0 / (self.weights ** 2).sum()
        
        # Resample if effective sample size is too low
        if eff_sample_size < self.n_particles / 2:
            self.resample()
    
    def resample(self):
        """
        Resample particles according to their weights using systematic resampling.
        """
        # Compute cumulative sum of weights
        cumsum = torch.cumsum(self.weights, dim=0)
        
        # Generate systematic resampling points
        u = torch.rand(1)
        positions = (u + torch.arange(self.n_particles)) / self.n_particles
        
        # Resample particles
        indices = torch.searchsorted(cumsum, positions)
        self.particles = self.particles[indices]
        
        # Reset weights to uniform
        self.weights = torch.ones(self.n_particles) / self.n_particles
    
    def estimate_posterior(self):
        """
        Compute posterior mean and variance from particles.
        
        Returns:
            tuple: (posterior_mean, posterior_variance)
        """
        posterior_mean = (self.particles.T * self.weights).sum(dim=1)
        posterior_var = ((self.particles - posterior_mean) ** 2).T * self.weights
        posterior_var = posterior_var.sum(dim=1)
        return posterior_mean.detach().numpy(), posterior_var.detach().numpy()
    
    def entropy(self):
        # """Compute the entropy of the variational distribution"""
        # posterior_mean = (self.particles.T * self.weights).sum(dim=1)
        # posterior_var = (((self.particles - posterior_mean) ** 2).T * self.weights).sum(dim=1)
    
        # q_dist = dist.Normal(posterior_mean, posterior_var.sqrt())
        # return q_dist.entropy().sum().detach().numpy()
        return (self.weights*torch.log(self.weights)).sum().detach().numpy()




if __name__ == "__main__":
    # Test the BetaVariationalBayesianInference class
    def f(X, theta):
        return torch.sigmoid(X @ theta.unsqueeze(-1)).squeeze()

    # Generate synthetic data
    np.random.seed(0)
    torch.manual_seed(0)
    n = 1000
    X = torch.rand(n, 4)
    theta_true = torch.tensor([0.5, 0.1, -0.1, 0.5])
    y = dist.Bernoulli(f(X, theta_true)).sample()

    # Fit the model
    model = BetaVariationalBayesianInference(f, input_dim=2, latent_dim=4, debug=True)
    model.fit(X, y, n_epochs=100, batch_size=64, lr=0.1)

    # Get the posterior parameters
    posterior = model.get_posterior_params()
    print("Posterior parameters:")
    print(posterior)

    # Compute the entropy of the variational distribution
    entropy = model.entropy()
    print(f"Entropy of variational distribution: {entropy:.4f}")

    # Sample from the variational distribution
    theta_samples = model.sample_latent(n_samples=1000)
    print(f"Sampled theta: {theta_samples[:5]}")

    # Compute the mean and mode of the variational distribution
    mean = model.low + (model.high - model.low) * (model.q_alpha / (model.q_alpha + model.q_beta))
    mode = (model.q_alpha - 1) / (model.q_alpha + model.q_beta - 2)
    print(f"Estimated theta mean: {mean}")
    print(f"Estimated theta mode: {mode}")
