import gymnasium as gym 
import numpy as np
import matplotlib.pyplot as plt
import random
import itertools
import torch
from agents.dqn import DoubleDQNAgent # for typing only
import pygame

class GridEnvDeform(gym.Env):
    def __init__(self, maze, l0,h0,l1,h1):

        self.original_maze = maze
        self.original_maze_shape = maze.shape
 
        self.maze = maze
        self.maze_shape = maze.shape

        # list of possible actions
        self.actions = [0,1,2,3]
        # list of possible orientations
        self.orientations = [0,1,2,3]
        # list of possible observations
        self.obs = list(itertools.product([0,1], repeat=5))
        # list of possible deformations
        self.deformations = [(i,j) for i in range(l0,h0) for j in range(l1,h1)]
        
        # space in which every maze lives (is a 2d matrix)
        self.max_shape = self.original_maze.shape * np.array([h1-1,h0-1]) + np.array([2,2])
        # list of states
        self.states = [((x,y,phi),(i,j)) for x in range(1,self.max_shape[0]-1) for y in range(1,self.max_shape[1]-1) for phi in range(4) for i in range(l0,h0) for j in range(l1,h1)] 
        self.l0 = l0
        self.h0 = h0
        self.l1 = l1
        self.h1 = h1

        self.goal_pos = self.original_maze.shape - np.array([2,2])
        
        self.set_rendering()

        self.reset()
        
    def T(self, s, a, s_):
        """ s = [(x, y, phi), thetas]
            T(s,a,s_) transition probability from s to s_ given action a 
            assuming is deterministic and always possible to move in the direction of the action (except at the boundaries)
        """
        # if the maze deformation is different the transition is impossible       
        if s[1] != s_[1]:
            return 0.
                
        return 1. if np.all(self.step(a,s)[0] == s_) else 0

    def O(self, s, a, o):
        # observation probability

        return 1 if np.all(self.get_observation(s) == o) else 0
    
    def R(self, s, a, s_=None):
        
        """R(s,a,s_) reward for getting to s_ (independent from previous state and action taken)"""
            
        return self.step(a,s)[1]            
    
    def step(self, a, s=None, execute=False):

        """take action a from state s (if given) or from actual state of the maze 
        
        return the next state, the reward, if the episode is terminated, if the episode is truncated, info"""
        
        if s is not None:
            self.set_state(s)
            x, y, phi = s[0][:3]
            x_, y_, phi_ = x, y, phi
        else:
            x, y = self.agent_pos
            phi = self.agent_orientation
            x_, y_, phi_ = x, y, phi

        actual_action = (a + phi) % 4
        
        if actual_action == 0:  # Move up
            new_pos = [x - 1, y]
        elif actual_action == 2:  # Move down
            new_pos = [x + 1, y]
        elif actual_action == 3:  # Move left
            new_pos = [x, y - 1]
        elif actual_action == 1:  # Move right
            new_pos = [x, y + 1]
        else:
            raise ValueError("Invalid Action")
        
        # Check if the new position is valid (inside the maze and not a wall)
        if 0 < new_pos[0] < self.max_shape[0]-1 and 0 < new_pos[1] < self.max_shape[1]-1:
            x_, y_ = new_pos
            if execute:
                self.agent_pos = new_pos

        phi_ = (phi + a) % 4
        
        if execute:
            self.agent_orientation = phi_
        
        terminated = np.all((x_,y_) == self.goal_pos)

        if np.all((x_,y_) == self.goal_pos):
            # if the agent is in the goal position
            reward =  1            
        elif np.all((x_,y_) == (x,y)):
            # if the agent has not moved (only at the boundary of the maze)
            reward =  -2 # -50/(self.max_shape[0]*self.max_shape[1])
        elif self.maze[x_, y_] == 1:
            # if the agent has entered a wall
            reward =  -2 # -50/(self.max_shape[0]*self.max_shape[1])
        elif self.maze[x_, y_] == 0:
            # if the agent has moved to a free cell
            reward =  -0.5 # -1/(self.max_shape[0]*self.max_shape[1])

        info = {}
        truncated = False 

        s_ = ((x_, y_, phi_), self.theta)
        
        if execute:
            self.timestep += 1
        
        return s_, reward, terminated, truncated, info, 
    
    def next_state(self, a, s, execute=False):

        """take action a from state s = x,y,phi 
        
        return the next state,
        """
    
        x, y, phi = s  
        x_, y_, phi_ = x, y, phi      

        actual_action = (a + phi) % 4
        
        if actual_action == 0:  # Move up
            new_pos = [x - 1, y]
        elif actual_action == 2:  # Move down
            new_pos = [x + 1, y]
        elif actual_action == 3:  # Move left
            new_pos = [x, y - 1]
        elif actual_action == 1:  # Move right
            new_pos = [x, y + 1]
        else:
            raise ValueError("Invalid Action")
        
        # Check if the new position is valid (inside the maze and not a wall)
        if 0 < new_pos[0] < self.max_shape[0]-1 and 0 < new_pos[1] < self.max_shape[1]-1:
            x_, y_ = new_pos

        phi_ = (phi + a) % 4
        
        s_ = (x_, y_, phi_)
        
        
        return s_ 

    def get_observation(self, s=None):

        if s is None:
            agent_pos = self.agent_pos
            agent_orientation = self.agent_orientation
        else:
            prior_state = self.get_state()
            self.set_deformed_maze(s[1])
            agent_pos = s[0][:2]
            agent_orientation = s[0][2]

        ind = [agent_pos + a for a in [np.array([0,-1]),
                                            np.array([-1,-1]),
                                            np.array([-1,0]),
                                            np.array([-1,+1]),
                                            np.array([0,+1]),
                                            np.array([+1,+1]),
                                            np.array([+1,0]),
                                            np.array([+1,-1])]]

        agent_obs = np.array([self.maze[tuple(ind[i%8])] 
                                                for i in range(2*agent_orientation, 2*agent_orientation+5)])
        
        if s is not None:
            self.set_state(prior_state)

        
        return agent_obs
    
    def set_deformed_maze(self,thetas: tuple):
        self.theta = thetas
        self.maze = self.stretch_maze(thetas)
        # self.goal_pos = self.maze.shape - np.array([thetas[1],thetas[0]])
        self.goal_pos = self.original_maze.shape * np.array([thetas[1],thetas[0]])

        canva1 = np.ones(self.max_shape, dtype=int)  # Start with walls
        # Place the original maze in the canvas
        canva1[1:self.maze.shape[0] + 1, 1:self.maze.shape[1] + 1] = self.maze

        self.maze = canva1
   
    def stretch_maze(self, thetas):
        scale_x, scale_y = thetas
        maze = self.original_maze

        original_height, original_width = maze.shape
        # Calculate new dimensions
        new_height = original_height * scale_y
        new_width = original_width * scale_x
        
        # Create a new maze with stretched dimensions
        stretched_maze = np.ones((new_height, new_width), dtype=int)

        # Fill the new maze with values from the original maze
        for i in range(original_height):
            for j in range(original_width):
                if maze[i, j] == 0:  # Path cell
                    # Fill the corresponding region in the stretched maze
                    stretched_maze[i*scale_y:(i+1)*scale_y, j*scale_x:(j+1)*scale_x] = 0

        return stretched_maze
    
    def set_state(self, s):
        theta0, theta1 = s[1][0], s[1][1]
        self.theta = (theta0, theta1)
        self.agent_pos = np.array(s[0][:2]) 
        self.agent_orientation = s[0][2]
        self.set_deformed_maze(s[1])
    
    def get_state(self):
        return (self.agent_pos[0],self.agent_pos[1], self.agent_orientation), self.theta
    
    def render(self, s=None, s_prime=None):
        """
        belief will be always distributed as follows 
        there exist a tuple (x_0,y_0,phi_0) such that
        for the two possible deformations theta0 theta1

        b((x_0,y_0,phi_0), theta_0) = b((x_0,y_0,phi_0), theta_1) = 0.5
        and 
        b = 0 everywere else
        
        rendering will show the same position in the two different possibile maze deformations
        along with the probability of each status
        """
        if s is not None:
            self.set_state(s)
        
        maze_render = np.copy(self.maze)
        maze_render[tuple(self.agent_pos)] = 2  # Show agent position
        maze_render[tuple(self.goal_pos)] = 4  # Show goal position
        plt.imshow(maze_render, cmap='binary', origin='upper')
        plt.show()

    def set_rendering(self):
        self.screen_width = 800
        self.screen_height = 600
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption("Maze Environment")            

    def render_bis(self, s=None):
        if s is not None:
            self.set_state(s)

        # Clear the screen
        self.screen.fill((255, 255, 255))

        # Draw the maze
        cell_size = min(self.screen_width, self.screen_height) // max(self.max_shape)
        for x in range(self.max_shape[0]):
            for y in range(self.max_shape[1]):
                if (x, y) == tuple(self.agent_pos):
                    color = (255, 0, 0)  # Red for agent
                elif (x, y) == tuple(self.goal_pos):
                    color = (0, 255, 0)  # Green for goal
                elif self.maze[x, y] == 1:
                    color = (0, 0, 0)  # Black for walls
                else:
                    color = (255, 255, 255)  # White for free space
                pygame.draw.rect(self.screen, color, (y * cell_size, x * cell_size, cell_size, cell_size))

        # Update the display
        pygame.display.flip()
   
    def close_render(self):
        pygame.quit()

    def reset(self, seed=42):
        randomdeformation = random.choice(self.deformations)
        self.agent_pos = [np.random.randint(1, self.max_shape[0]-1), np.random.randint(1, self.max_shape[1]-1)]
        self.agent_orientation = random.choice(self.orientations)
        self.set_deformed_maze(randomdeformation)
        self.goal_pos = self.original_maze.shape * np.array([randomdeformation[1],randomdeformation[0]])
        self.theta = randomdeformation
        self.timestep = 0
        return ((self.agent_pos[0],self.agent_pos[1], self.agent_orientation), self.theta), {}
        
    def is_done(self):
        return np.all(self.agent_pos == self.goal_pos)
    
    def is_new(self):
        return self.timestep == 0

class POMDPWrapper_v0():
    """This is a wrapper for the GridEnvDeform class that makes the environment partially observable."""

    def __init__(self, env: GridEnvDeform, agent : DoubleDQNAgent, T,R,O,*args, **kwargs):

        self.agent = agent
        self.states = [((x,y,phi),(i,j)) for x in range(1,env.max_shape[0]-1) for y in range(1,env.max_shape[1]-1) for phi in range(4) for i in range(env.l0,env.h0) for j in range(env.l1,env.h1)] 
        self.actions = [0,1,2,3]
        self.observations = list(itertools.product([0,1], repeat=5))
        self.thetas = env.deformations

        self.obs_dict = {obs : i for i, obs in enumerate(self.observations)}
        self.state_dict = {state : i for i, state in enumerate(self.states)}
        
        # Transition, Observation and Reward model T(S,A,S'), O(S,A,O), R(S,A,S')
        self.T = T
        self.O = O
        self.R = R
    
    def step(self,s,a):
        s_prime = torch.argmax(self.T[s,a,:])
        r = self.R[s,a,s_prime.item()]
        obs = torch.argmax(self.O[s_prime,0,:])
        info = {"actual_state": s_prime.item()}

        # done = True if np.all(self.states[s_prime.item()][0][:2] == self.env.goal_pos) else False
        done = True if r.item() == 10 else False

        return obs.item(), r.item(), done , info
    
    def run(self, num_trajectories):
        """ 
        The idea is to run the environment to populate the replay buffer with some data
        transition for the POMDP should be like (belief, action, reward, next_belief, done)
        
        repeat
        
        Initialize belief if new episode
        for all actions compute b,a,b',r and store
        execute the best action in the environment
        get new belief 
        
        until populated replay buffer
        
        """


        # create a list of trajectories
        trajectories = []

        # Initialize belief (uniform distribution for now, probably need to change this)
        b = (torch.ones(len(self.states)) / len(self.states))
        s = np.random.randint(0,len(self.states))

        while len(trajectories) < num_trajectories:
            for a in self.actions:
                next_obs, reward, done, info = self.step(s, a)
                b_prime = self.update_belief(b, a, next_obs)

                # POMDP feed the agent with the belief state
                #state = {'obs':b, 'raw_legal_actions':la, 'legal_actions':la}
                #next_state = {'obs':b_prime, 'raw_legal_actions':la, 'legal_actions':la}
                
                # Store trajectory
                # trajectories.append(({'obs':b}, a, reward, {'obs':b_prime}, done))
                trajectories.append((b, a, reward, b_prime, done))

            # step in the environment
            best_action = self.agent.step(b)    
            next_obs, _, done, info = self.step(s,best_action)
            b = self.update_belief(b, best_action, next_obs)
            s = info['actual_state']
            # if done:
            #     s = np.random.randint(0,len(self.states))
            #     b = (torch.ones(len(self.states)) / len(self.states))
        
        return trajectories

    def reset(self):
        # set random state
        s = np.random.randint(0,len(self.states))

        # get observation
        obs = torch.argmax(self.O[s,0])
        info = {"actual_state": s}
        return obs, info

    def update_belief(self, belief, action, observation):
        """
        Perform a Bayesian belief update in a POMDP with action-dependent transition and observation models.

        Parameters:
            belief (torch.Tensor): Initial belief distribution over states, shape (num_states,)
            T (torch.Tensor): Transition probabilities, shape (num_states, num_actions, num_states)
            O (torch.Tensor): Observation probabilities, shape (num_states, num_actions, num_observations)
            action (int): The action taken (index of action)
            observation (int): The observation received (index of observation)

        Returns:
            torch.Tensor: The updated belief over states, shape (num_states,)
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
 
class POMDPWrapper_v1():
    """This is a wrapper for the GridEnvDeform class that makes the environment partially observable."""

    def __init__(self, env: GridEnvDeform, T,O,R,*args, **kwargs):

        self.env = env
        self.states = [((x,y,phi),(i,j)) for x in range(1,env.max_shape[0]-1) for y in range(1,env.max_shape[1]-1) for phi in range(4) for i in range(env.l0,env.h0) for j in range(env.l1,env.h1)] 
        self.actions = [0,1,2,3]
        self.observations = list(itertools.product([0,1], repeat=5))
        self.thetas = env.deformations

        self.obs_dict = {obs : i for i, obs in enumerate(self.observations)}
        self.state_dict = {state : i for i, state in enumerate(self.states)}
        
        # Transition, Observation and Reward model T(S,A,S'), O(S,A,O), R(S,A,S')
        self.T = T
        self.O = O
        self.R = R
    
    def step(self, s,a):
        s_prime = torch.argmax(self.T[s,a,:])
        r = self.R[s,a,s_prime]
        obs = torch.argmax(self.O[s_prime,0,:])
        info = {"actual_state": s_prime.item()}

        # done = True if np.all(self.states[s_prime.item()][0][:2] == self.env.goal_pos) else False
        done = True if r.item() == 10 else False
        
        return obs.item(), r.item(), done , info
    
    def reset(self):
        return np.random.randint(0,len(self.states)), {}

    def update_belief(self, belief, action, observation):
        """
        Perform a Bayesian belief update in a POMDP with action-dependent transition and observation models.

        Parameters:
            belief (torch.Tensor): Initial belief distribution over states, shape (num_states,)
            T (torch.Tensor): Transition probabilities, shape (num_states, num_actions, num_states)
            O (torch.Tensor): Observation probabilities, shape (num_states, num_actions, num_observations)
            action (int): The action taken (index of action)
            observation (int): The observation received (index of observation)

        Returns:
            torch.Tensor: The updated belief over states, shape (num_states,)
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


def create_maze(dim):
    maze = np.ones((dim*2+1, dim*2+1))
    x, y = (0, 0)
    maze[2*x+1, 2*y+1] = 0
    stack = [(x, y)]
    
    while len(stack) > 0:
        x, y = stack[-1]
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        random.shuffle(directions)

        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if nx >= 0 and ny >= 0 and nx < dim and ny < dim and maze[2*nx+1, 2*ny+1] == 1:
                maze[2*nx+1, 2*ny+1] = 0
                maze[2*x+1+dx, 2*y+1+dy] = 0
                stack.append((nx, ny))
                break
        else:
            stack.pop()

    # Create entrance and exit
    # maze[1, 0] = 0
    # maze[-2, -1] = 0

    return maze[1:-1, 1:-1]
