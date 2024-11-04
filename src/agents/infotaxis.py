import torch

class Infotaxis():
  
    def __init__(self, T, O, R):
        self.T = T
        self.O = O
        self.R = R
    
    def get_entropy(self, probabilities):
        
        # Calculate entropy, avoiding log(0) by adding a mask
        entropy = -torch.sum(probabilities * torch.log2(probabilities + (probabilities == 0).float()))
        
        return entropy.item()  # .item() to get a standard Python float

    def update_belief(self, belief, action, observation):
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

            P(s'|b,a,o) = \eta * P(o|s',a) sum_s P(s'|s,a)b(s) = \eta P(o|s',a) P(s'|b,a) 
        """
        # Prediction Step: Compute predicted belief over next states
        predicted_belief = torch.matmul(belief, self.T[:, action])

        # Update Step: Multiply by observation likelihood
        observation_likelihood = self.O[:, action, observation]
        new_belief = predicted_belief * observation_likelihood

        # Normalize the updated belief to ensure it's a valid probability distribution
        if new_belief.sum() > 0:
            new_belief /= new_belief.sum() 
                
        return new_belief
    
    def update_belief_superbatched(self, belief):
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
        # [predicted_belief]_a,s' = sum_s P(s'|s,a) * [belief]_s
        predicted_belief = torch.einsum('s,san->an', belief, self.T)

        # Update Step: Multiply by observation likelihood
        new_belief = torch.einsum('as,sao->sao', predicted_belief, self.O)


        # Normalize the updated belief to ensure it's a valid probability distribution
        new_belief /= new_belief.sum(dim=0) + 1e-10
                
        return new_belief

    def G(self,b,a=None):
        """"
        
        input:
        b: belief
        a: action

        output: 
        G: torch.tensor shape (numactions) expected reduction of the belief's entropy G (b, a)
        
        
        G(b,a) = H(b) - sum_o P(o|b,a)H(b|a,o)
        P(o|b,a) = sum_s P(o|s,a)P(s|b,a) = sum_s P(o|s,a) sum_s' P(s|s',a)P(s'|b)
        
        """
        # H(b) = entropy of belief
        H_b = self.get_entropy(b)
        
        # P_o_b_a = torch.matmul(b, self.O[:,a,:])
        # [P_o_b_a]_o_a = P(o|b,a) = sum_s P(o|s,a)P(s|b,a) = sum_s P(o|s,a) sum_s'        shape (numobservations,numactions)
        P_o_b_a = torch.einsum('sao,nas,n->oa',self.O,self.T,b)

        # [bprime]_s_a_o = P(s|b,a,o) shape (numstates, numactions, numobservations)
        brpime = self.update_belief_superbatched(b)

        # [H_b_a_o]_i = H(b|a,o_i) = entropy of updated belief shape (numactions, numobservations)
        H_b_a_o = torch.sum(brpime * torch.log2(brpime + (brpime == 0).float()), dim=0)

        G = H_b - torch.einsum('oa,ao->a', P_o_b_a, H_b_a_o)

        if a is not None:
            return G[a]
        
        return G

    def get_action(self, belief):
        """
        Compute the action that minimizes the expected entropy of the belief.

        Parameters:
            belief (torch.Tensor): The current belief over states, shape (num_states,)

        Returns:
            int: The action that minimizes the expected entropy of the belief
        """
        # Compute the expected entropy reduction for each action
        G = self.G(belief)
        
        # Return the action that minimizes the expected entropy
        return torch.argmin(G).item()
        