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
