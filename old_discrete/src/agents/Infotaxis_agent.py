import torch
import numpy as np
from environment.env import GridEnvDeform


class ThetaInfotaxis():
  
    def __init__(self, env: GridEnvDeform, obs_model=None):
        self.env = env    
        if obs_model is not None:
            self.obs_model = obs_model
            self.obs_model.eval()
            self.update_belief = self.update_belief_superbatched
        
    def get_entropy(self, probabilities):
        
        # Calculate entropy, avoiding log(0) by adding a mask
        entropy = -torch.sum(probabilities * torch.log2(probabilities + (probabilities == 0).float()))
        
        return entropy.item()  # .item() to get a standard Python float

    def update_belief(self, belief, pos, observation):
        """"
        perform update over theta
        
        $$b'_{x,a,o}(theta) = \eta \cdot p(o|x,theta) \cdot b(theta)$$
        
        """

        new_belief = torch.zeros_like(belief)

        for t, theta in enumerate(self.env.deformations):
            P_o_s_theta = np.all(self.env.get_observation(s = (pos,theta)) == observation) # 0 or 1 

            new_belief[t] = P_o_s_theta * belief[t]
        
        new_belief = new_belief / (torch.sum(new_belief) + 1e-10)

        return new_belief
    
    def update_belief_superbatched(self, belief, pos, observation):
        """"
        perform update over theta
        """
        thetas = torch.tensor(self.env.deformations)
        obs = torch.tensor(observation, dtype=torch.float32)
        pos = torch.tensor(pos, dtype=torch.float32)

        B = thetas.shape[0]
        
        P_o_s_theta = torch.distributions.Bernoulli(self.obs_model(pos.expand(B,-1),thetas)).log_prob(obs.expand(B,-1)).sum(dim=1)

        belief = torch.exp(P_o_s_theta) * belief

        belief = belief/belief.sum()

        
        return belief

    def G(self,b,pos):
        """"
        
        input:
        b: belief
        a: action

        output: 
        G: torch.tensor shape (numactions) expected reduction of the belief's entropy G (b, a)
        
        
        G(b,a) = H(b) - sum_o P(o|b,a)H(b|a,o)
        P(o|b,a) = sum_s P(o|s,a)P(s|b,a) = sum_theta P(o|s=(pos,theta),a)b(theta)
        
        """
        # H(b) = entropy of belief
        H_b = self.get_entropy(b)
                
        # this is P_o_b_a = P(o|b) = sum_s P(o|s=(pos,theta)) * b(theta) = sum_theta P(o|s=(pos,theta)) * b(theta) 
        # P_o_x_t]_o_t = P(o|s=(pos,t)) shape (numobservations,)
        P_o_x_t = torch.zeros((32,))
        for o, obs in enumerate(self.env.obs):
            for t, theta in enumerate(self.env.deformations):
                P_o_x_t[o] += 1*np.all(self.env.get_observation(s = (pos,theta)) == obs) * b[t]


        # updated_belief b_a_o = b(theta) * P(o|s=(pos+a,theta))
        # b_a_o]_a_o_t = b'_a_o(t) shape (numactions,numobservations,numdeformations)
        b_a_o = torch.zeros((len(self.env.actions),len(self.env.obs), len(self.env.deformations)))
        for a, action in enumerate(self.env.actions):
            next_pos = self.env.next_state(a,pos)
            for o, obs in enumerate(self.env.obs):
                b_a_o[a][o] = self.update_belief(b,next_pos,obs)

        # # [H_b_a_o]_i = H(b|a,o_i) = entropy of updated belief shape (numactions, numobservations)
        H_b_a_o = -torch.sum(b_a_o * torch.log2(b_a_o + (b_a_o == 0).float()), dim=2)

        G = H_b - torch.einsum('o,ao->a', P_o_x_t, H_b_a_o)

        return G

    def get_action(self, belief, pos):
        """
        Compute the action that minimizes the expected entropy of the belief.

        Parameters:
            belief (torch.Tensor): The current belief over states, shape (num_states,)

        Returns:
            int: The action that minimizes the expected entropy of the belief
        """
        # Compute the expected entropy reduction for each action
        G = self.G(belief,pos)
        print(G)
        print(torch.argmax(G))
                      
        # Return the action that minimizes the expected entropy
        return torch.argmax(G).item()


class IDS():
  
    def __init__(self, env: GridEnvDeform, obs_model = None):
        self.env = env
        if obs_model is not None:
            self.obs_model = obs_model
            self.obs_model.eval()
            self.update_belief = self.update_belief_superbatched

    def set_belief(self,belief):
        self.belief = belief

    def get_entropy(self, probabilities):
        
        # Calculate entropy, avoiding log(0) by adding a mask
        entropy = -torch.sum(probabilities * torch.log2(probabilities + (probabilities == 0).float()))
        
        return entropy.item()  # .item() to get a standard Python float

    def update_belief(self, belief, pos, observation):
        """"
        perform update over theta
        
        $$b'_{x,a,o}(theta) = \eta \cdot p(o|x,theta) \cdot b(theta)$$
        
        """

        new_belief = torch.zeros_like(belief)

        for t, theta in enumerate(self.env.deformations):
            P_o_s_theta = np.all(self.env.get_observation(s = (pos,theta)) == observation) # 0 or 1 

            new_belief[t] = P_o_s_theta * belief[t]
        
        new_belief = new_belief / (torch.sum(new_belief) + 1e-10)

        return new_belief
    
    def update_belief_superbatched(self, belief, pos, observation):
        """"
        perform update over theta
        """
        thetas = torch.tensor(self.env.deformations)
        obs = torch.tensor(observation, dtype=torch.float32)
        pos = torch.tensor(pos, dtype=torch.float32)

        B = thetas.shape[0]
        
        with torch.no_grad():
            P_o_s_theta = torch.distributions.Bernoulli(self.obs_model(pos.expand(B,-1),thetas)).log_prob(obs.expand(B,-1)).sum(dim=1)

        belief = belief* torch.exp(P_o_s_theta)

        belief = belief/belief.sum()

        return belief

    def G(self,b,pos):
        """"
        
        input:
        b: belief
        pos: actual position (known)
        
        output: 
        G: torch.tensor shape (numactions) expected reduction of the belief's entropy G(b, a)
        
        
        G(b,a) = H(b) - sum_o P(o|b,a)H(b|a,o)
        P(o|b,a) = sum_s P(o|s,a)P(s|b,a) = sum_theta P(o|s=(pos,theta),a)b(theta)
        
        """
        # H(b) = entropy of belief
        H_b = self.get_entropy(b)
                
        # this is P_o_b_a = P(o|b) = sum_s P(o|s=(pos,theta)) * b(theta) = sum_theta P(o|s=(pos,theta)) * b(theta) 
        # P_o_x_t]_o_t = P(o|s=(pos,t)) shape (numobservations,)
        P_o_x_t = torch.zeros((32,))
        for o, obs in enumerate(self.env.obs):
            for t, theta in enumerate(self.env.deformations):
                P_o_x_t[o] += 1*np.all(self.env.get_observation(s = (pos,theta)) == obs) * b[t]


        # updated_belief b_a_o = b(theta) * P(o|s=(pos+a,theta))
        # b_a_o]_a_o_t = b'_a_o(t) shape (numactions,numobservations,numdeformations)
        b_a_o = torch.zeros((len(self.env.actions),len(self.env.obs), len(self.env.deformations)))
        for a, action in enumerate(self.env.actions):
            next_pos = self.env.next_state(a,pos)
            for o, obs in enumerate(self.env.obs):
                b_a_o[a][o] = self.update_belief(b,next_pos,obs)

        # # [H_b_a_o]_i = H(b|a,o_i) = entropy of updated belief shape (numactions, numobservations)
        H_b_a_o = -torch.sum(b_a_o * torch.log2(b_a_o + (b_a_o == 0).float()), dim=2)

        G = H_b - torch.einsum('o,ao->a', P_o_x_t, H_b_a_o)

        return G

    def Delta(self,b,pos):
        """compute expected regret for each action
        
        \Delta = max_a(\sum_s R(s,a)b(s))-\sum_s R(s,a)b(s)
        """

        # R(s,a) = 1 if s is the goal state, 0 otherwise
        R_t_a = torch.zeros(len(self.env.deformations),len(self.env.actions))
        for t, theta in enumerate(self.env.deformations):
            for a, action in enumerate(self.env.actions):
                R_t_a[t][a] = self.env.R((pos,theta),a)
        
        E_R = torch.einsum('ta,t->a',R_t_a,b)

        # expected regret for each action
        Delta = torch.max(E_R) - E_R

        return Delta

    def get_action(self, belief,pos):
        """
        Compute the distribution over actions that minimizes ratio of the expected reduction of the belief's entropy and the expected regret
        
        pi = argmin_pi (delta(pi) / G(pi))
        where
        delta(pi) = sum_a pi(a) * delta(a)
        G(pi) = sum_a pi(a) * G(a)
        """
        
        G_a = self.G(belief,pos)
        Delta_a = self.Delta(belief,pos)

        q = torch.zeros((len(self.env.actions),len(self.env.actions)))
        for a in range(len(self.env.actions)):
            for aa in range(a, len(self.env.actions)):                
                q[a][aa] = self._q_a_aa(a,aa, Delta_a,G_a) 

        tmp = self._aastar(q,Delta_a,G_a)        
        astar, astarstar = tmp[0], tmp[1]
        
        # sample form a bernoulli distribution with parameter q
        b = torch.bernoulli(q[astar,astarstar])        
        return (b*astar + (1-b)*astarstar).item()

    def _q_a_aa(self,a,aa,Delta,G):
        """compute the argmin"""

        q = torch.tensor(np.linspace(0, 1, 100))
        ratio = torch.square(q*Delta[a] + (1-q) * Delta[aa]) / (q*G[a] + (1-q) * G[aa])
        return q[np.argmin(ratio)]
    
    def _aastar(self,q,Delta,G):
        """compute the argmin"""

        ratio = torch.zeros((len(self.env.actions),len(self.env.actions)))
        for a in range(len(self.env.actions)):
            for aa in range(a, len(self.env.actions)):

                ratio[a,aa] = torch.square(q[a,aa]*Delta[a] + (1-q[a,aa]) * Delta[aa]) / (q[a,aa]*G[a] + (1-q[a,aa]) * G[aa])

        minindex = torch.argmin(ratio)
        return torch.unravel_index(minindex, ratio.shape)
    




# class Infotaxis():
  
#     def __init__(self, T, O, R):
#         self.T = T
#         self.O = O
#         self.R = R
    
#     def get_entropy(self, probabilities):
        
#         # Calculate entropy, avoiding log(0) by adding a mask
#         entropy = -torch.sum(probabilities * torch.log2(probabilities + (probabilities == 0).float()))
        
#         return entropy.item()  # .item() to get a standard Python float

#     def update_belief(self, belief, action, observation):
#         """
#         Perform a Bayesian belief update in a POMDP with action-dependent transition and observation models.

#         Parameters:
#             belief (torch.Tensor): Initial belief distribution over states, shape (num_states,)
#             action (int): The action taken (index of action)
#             observation (int): The observation received (index of observation)
#             T (torch.Tensor): Transition probabilities, shape (num_states, num_actions, num_states)
#             O (torch.Tensor): Observation probabilities, shape (num_states, num_actions, num_observations)

#         Returns:
#             torch.Tensor: The updated belief over states, shape (num_states,)

#             P(s'|b,a,o) = \eta * P(o|s',a) sum_s P(s'|s,a)b(s) = \eta P(o|s',a) P(s'|b,a) 
#         """
#         # Prediction Step: Compute predicted belief over next states
#         predicted_belief = torch.matmul(belief, self.T[:, action])

#         # Update Step: Multiply by observation likelihood
#         observation_likelihood = self.O[:, action, observation]
#         new_belief = predicted_belief * observation_likelihood

#         # Normalize the updated belief to ensure it's a valid probability distribution
#         if new_belief.sum() > 0:
#             new_belief /= new_belief.sum() 
                
#         return new_belief
    
#     def update_belief_superbatched(self, belief):
#         """
#         Perform a Bayesian belief update in a POMDP with action-dependent transition and observation models.
#         Returns the updated belief over states for every action and observation combo (a 3d tensor).

#         Parameters:
#             belief (torch.Tensor): Initial belief distribution over states, shape (num_states,)
#             T (torch.Tensor): Transition probabilities, shape (num_states, num_actions, num_states)
#             O (torch.Tensor): Observation probabilities, shape (num_states, num_actions, num_observations)

#         Returns:
#             torch.Tensor: The updated belief over states, shape (num_states,numactions,numobservations)
#         """
#         # Prediction Step: Compute predicted belief over next states
#         # [predicted_belief]_a,s' = sum_s P(s'|s,a) * [belief]_s
#         predicted_belief = torch.einsum('s,san->an', belief, self.T)

#         # Update Step: Multiply by observation likelihood
#         new_belief = torch.einsum('as,sao->sao', predicted_belief, self.O)


#         # Normalize the updated belief to ensure it's a valid probability distribution
#         new_belief /= new_belief.sum(dim=0) + 1e-10
                
#         return new_belief

#     def G(self,b,a=None):
#         """"
        
#         input:
#         b: belief
#         a: action

#         output: 
#         G: torch.tensor shape (numactions) expected reduction of the belief's entropy G (b, a)
        
        
#         G(b,a) = H(b) - sum_o P(o|b,a)H(b|a,o)
#         P(o|b,a) = sum_s P(o|s,a)P(s|b,a) = sum_s P(o|s,a) sum_s' P(s|s',a)P(s'|b)
        
#         """
#         # H(b) = entropy of belief
#         H_b = self.get_entropy(b)
        
#         # P_o_b_a = torch.matmul(b, self.O[:,a,:])
#         # [P_o_b_a]_o_a = P(o|b,a) = sum_s P(o|s,a)P(s|b,a) = sum_s P(o|s,a) sum_s'        shape (numobservations,numactions)
#         P_o_b_a = torch.einsum('sao,nas,n->oa',self.O,self.T,b)

#         # [bprime]_s_a_o = P(s|b,a,o) shape (numstates, numactions, numobservations)
#         brpime = self.update_belief_superbatched(b)

#         # [H_b_a_o]_i = H(b|a,o_i) = entropy of updated belief shape (numactions, numobservations)
#         H_b_a_o = torch.sum(brpime * torch.log2(brpime + (brpime == 0).float()), dim=0)

#         G = H_b - torch.einsum('oa,ao->a', P_o_b_a, H_b_a_o)

#         if a is not None:
#             return G[a]
        
#         return G

#     def get_action(self, belief):
#         """
#         Compute the action that minimizes the expected entropy of the belief.

#         Parameters:
#             belief (torch.Tensor): The current belief over states, shape (num_states,)

#         Returns:
#             int: The action that minimizes the expected entropy of the belief
#         """
#         # Compute the expected entropy reduction for each action
#         G = self.G(belief)
        
#         # Return the action that minimizes the expected entropy
#         return torch.argmin(G).item()


# class SpaceAwareInfotaxis():
  
#     def __init__(self, T, O, R, distances):
#         self.T = T
#         self.O = O
#         self.R = R
#         self.distances = distances 
    
#     def get_entropy(self, probabilities):
        
#         # Calculate entropy, avoiding log(0) by adding a mask
#         entropy = -torch.sum(probabilities * torch.log2(probabilities + (probabilities == 0).float()))
        
#         return entropy.item()  # .item() to get a standard Python float

#     def update_belief(self, belief, action, observation):
#         """
#         Perform a Bayesian belief update in a POMDP with action-dependent transition and observation models.

#         Parameters:
#             belief (torch.Tensor): Initial belief distribution over states, shape (num_states,)
#             action (int): The action taken (index of action)
#             observation (int): The observation received (index of observation)
#             T (torch.Tensor): Transition probabilities, shape (num_states, num_actions, num_states)
#             O (torch.Tensor): Observation probabilities, shape (num_states, num_actions, num_observations)

#         Returns:
#             torch.Tensor: The updated belief over states, shape (num_states,)

#             P(s'|b,a,o) = \eta * P(o|s',a) sum_s P(s'|s,a)b(s) = \eta P(o|s',a) P(s'|b,a) 
#         """
#         # Prediction Step: Compute predicted belief over next states
#         predicted_belief = torch.matmul(belief, self.T[:, action])

#         # Update Step: Multiply by observation likelihood
#         observation_likelihood = self.O[:, action, observation]
#         new_belief = predicted_belief * observation_likelihood

#         # Normalize the updated belief to ensure it's a valid probability distribution
#         if new_belief.sum() > 0:
#             new_belief /= new_belief.sum() 
                
#         return new_belief
    
#     def update_belief_superbatched(self, belief):
#         """
#         Perform a Bayesian belief update in a POMDP with action-dependent transition and observation models.
#         Returns the updated belief over states for every action and observation combo (a 3d tensor).

#         Parameters:
#             belief (torch.Tensor): Initial belief distribution over states, shape (num_states,)
#             T (torch.Tensor): Transition probabilities, shape (num_states, num_actions, num_states)
#             O (torch.Tensor): Observation probabilities, shape (num_states, num_actions, num_observations)

#         Returns:
#             torch.Tensor: The updated belief over states, shape (num_states,numactions,numobservations)
#         """
#         # Prediction Step: Compute predicted belief over next states
#         # [predicted_belief]_a,s' = sum_s P(s'|s,a) * [belief]_s
#         predicted_belief = torch.einsum('s,san->an', belief, self.T)

#         # Update Step: Multiply by observation likelihood
#         new_belief = torch.einsum('as,sao->sao', predicted_belief, self.O)


#         # Normalize the updated belief to ensure it's a valid probability distribution
#         new_belief /= new_belief.sum(dim=0) + 1e-10
                
#         return new_belief

#     def G(self,b,a=None):
#         """"
        
#         input:
#         b: belief
#         a: action

#         output: 
#         G: torch.tensor shape (numactions) expected reduction of the belief's entropy G (b, a)
        
        
#         G(b,a) = H(b) - sum_o P(o|b,a)H(b|a,o)
#         P(o|b,a) = sum_s P(o|s,a)P(s|b,a) = sum_s P(o|s,a) sum_s' P(s|s',a)P(s'|b)
        
#         """
#         # H(b) = entropy of belief
#         H_b = self.get_entropy(b)
        
#         # P_o_b_a = torch.matmul(b, self.O[:,a,:])
#         # [P_o_b_a]_o_a = P(o|b,a) = sum_s P(o|s,a)P(s|b,a) = sum_s P(o|s,a) sum_s'        shape (numobservations,numactions)
#         P_o_b_a = torch.einsum('sao,nas,n->oa',self.O,self.T,b)

#         # [bprime]_s_a_o = P(s|b,a,o) shape (numstates, numactions, numobservations)
#         bprime = self.update_belief_superbatched(b)

#         # [H_b_a_o]_a_o = H(b|a,o) = entropy of updated belief shape (numactions, numobservations)
#         H_b_a_o = torch.sum(bprime * torch.log2(bprime + (bprime == 0).float()), dim=0)
#         e_H = torch.exp(H_b_a_o)


#         D_b_a_o = torch.einsum('sao,s->ao',bprime, self.distances)

#         G = H_b - torch.einsum('oa,ao->a', P_o_b_a, e_H - D_b_a_o)

#         if a is not None:
#             return G[a]
        
#         return G

#     def get_action(self, belief):
#         """
#         Compute the action that minimizes the expected entropy of the belief.

#         Parameters:
#             belief (torch.Tensor): The current belief over states, shape (num_states,)

#         Returns:
#             int: The action that minimizes the expected entropy of the belief
#         """
#         # Compute the expected entropy reduction for each action
#         G = self.G(belief)
        
#         # Return the action that minimizes the expected entropy
#         return torch.argmin(G).item()
        