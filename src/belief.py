import torch

def belief_entropy(probabilities):
    
    # Calculate entropy, avoiding log(0) by adding a mask
    entropy = -torch.sum(probabilities * torch.log2(probabilities + (probabilities == 0).float()))
    
    return entropy.item()  # .item() to get a standard Python float


def update_belief(self, belief, action, observation, T, O):
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
    observation_likelihood = O[:, action, self.obs_dict[tuple(observation.tolist())]]
    new_belief = predicted_belief * observation_likelihood

    # Normalize the updated belief to ensure it's a valid probability distribution
    if new_belief.sum() > 0:
        new_belief /= new_belief.sum() 
            
    return new_belief
