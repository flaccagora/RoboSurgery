import gymnasium as gym 
import numpy as np
import matplotlib.pyplot as plt
import random
import itertools
import torch
import pygame
import imageio
from gymnasium.spaces import Dict, Discrete, Box
from collections import OrderedDict
from utils.point_in import is_point_in_parallelogram, sample_in_parallelogram
from environment.cpp_env_continous import gridworld


###----------------------------DISCRETE----------------------------###
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

class GridEnvDeform(gym.Env):
    def __init__(self, maze, l0,h0,l1,h1, render_mode=None):

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
        self.state_dict = {state : i for i, state in enumerate(self.states)}
        
        self.l0 = l0
        self.h0 = h0
        self.l1 = l1
        self.h1 = h1

        self.goal_pos = self.original_maze.shape - np.array([2,2])
        
        self.frames = []
        self.render_mode = render_mode
        if self.render_mode == "human":
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
            self.episode.append(s_)

        if self.render_mode == "human":
            self.render_bis()

        
        return s_, reward, terminated, truncated, info
        
    def backward(self):
        """go back to previous state"""

        if self.timestep == 0:
            return "No previous state"
        
        s = self.episode[self.timestep]
        self.timestep -= 1
        self.set_state(s)

        return s
    
    def forward(self):
        """go forward to next state"""

        if self.timestep == len(self.episode):
            return "No next state"
        
        s = self.episode[self.timestep]
        self.timestep += 1
        self.set_state(s)  
        
        return s
         
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
        pygame.init()  # Initialize all pygame modules
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption("Maze Environment")
        
        # Handle key events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return
            elif event.type == pygame.KEYDOWN:
                # Press 'r' to reset environment
                if event.key == pygame.K_r:
                    self.reset()
                # Press 'q' to quit
                elif event.key == pygame.K_q:
                    pygame.quit()
                    return
                # Press 's' to save current state
                elif event.key == pygame.K_s:
                    self.save_state()
                # Press space to pause/resume
                elif event.key == pygame.K_SPACE:
                    self.pause()
                # Press arrow keys for manual control
                elif event.key == pygame.K_LEFT:
                    self.step(3)  # Left action
                elif event.key == pygame.K_RIGHT:
                    self.step(1)  # Right action
                elif event.key == pygame.K_UP:
                    self.step(0)  # Up action
                elif event.key == pygame.K_DOWN:
                    self.step(2)  # Down action

        # Update display
        pygame.display.flip()

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

        # Add text for controls
        font = pygame.font.Font(None, 36)
        controls = [
            "Controls:",
            "R - Reset",
            "Q - Quit",
            "Space - Pause/Resume",
            "Arrows - Move agent"
        ]
        
        for i, text in enumerate(controls):
            text_surface = font.render(text, True, (0, 0, 0))
            self.screen.blit(text_surface, (self.screen_width - 200, 20 + i * 30))

        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    self.reset()
                elif event.key == pygame.K_q:
                    pygame.quit()
                    return
                elif event.key == pygame.K_SPACE:
                    self.pause()
                elif event.key == pygame.K_LEFT:
                    self.step(3,execute=True)
                elif event.key == pygame.K_RIGHT:
                    self.step(1,execute=True)
                elif event.key == pygame.K_UP:
                    self.step(0,execute=True)
                elif event.key == pygame.K_DOWN:
                    self.step(2,execute=True)
                

        # Update the display
        pygame.display.flip()

        # Capture the current frame and add it to the list of frames
        frame = pygame.surfarray.array3d(self.screen)
        self.frames.append(frame)
    
    def save_gif(self):
        """
        Save the captured frames as a GIF file.
        """
        imageio.mimsave("gif.gif", self.frames, duration=0.1)
        print(f"GIF saved as gif")

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
        self.episode = [(self.get_state())]

        if self.render_mode == "human":
            self.render_bis()

        return ((self.agent_pos[0],self.agent_pos[1], self.agent_orientation), self.theta), {}
        
    def is_done(self):
        return np.all(self.agent_pos == self.goal_pos)
    
    def is_new(self):
        return self.timestep == 0

class POMDPGYMGridEnvDeform(gym.Env):
    
    def __init__(self, maze, l0,h0,l1,h1,render_mode = None):

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
        self.state_dict = {state : i for i, state in enumerate(self.states)}
        self.positions = [(x,y,phi) for x in range(1,self.max_shape[0]-1) for y in range(1,self.max_shape[1]-1) for phi in range(4)]
        
        self.l0 = l0
        self.h0 = h0
        self.l1 = l1
        self.h1 = h1

        self.goal_pos = self.original_maze.shape - np.array([2,2])
        
        self.frames = []
        self.reset()

        self.render_mode = render_mode
        if self.render_mode == "human":
            self.set_rendering()

        # gym attributes
        self.action_space = gym.spaces.Discrete(4)
        self.observation_space =  Dict({
                                    "x": Discrete(self.max_shape[0]),              # Values from 0 to 10
                                    "y": Discrete(self.max_shape[1]),              # Values from 0 to 10
                                    "phi": Discrete(5),             # Values from 0 to 4
                                    "belief": Box(low=0.0, high=1.0, shape=(len(self.deformations),), dtype=float)  # Probability vector
                                })
    
    def step(self, a):

        """take action a from state s (if given) or from actual state of the maze 
        
        return the next state, the reward, if the episode is terminated, if the episode is truncated, info"""
        
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
            self.agent_pos = new_pos

        phi_ = (phi + a) % 4
        
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
        
        self.timestep += 1
        truncated = self.timestep >= 100 
                

        if self.render_mode == "human":
            self.render()

        new_beleif = self.update_belief()

        self.belief = new_beleif

        obs = OrderedDict({
                            "x": torch.tensor([x_],dtype=torch.int32),              # Values from 0 to 10
                            "y": torch.tensor([y_],dtype=torch.int32),              # Values from 0 to 10
                            "phi": torch.tensor([phi_],dtype=torch.int32),             # Values from 0 to 4
                            "belief": self.belief , # Probability vector
                        })

        
        return obs, reward, terminated, truncated, info
    
    def set_render_mode(self, mode):
        self.render_mode = mode
        if self.render_mode == "human":
            self.set_rendering()
        
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

    def set_state(self, s):
        theta0, theta1 = s[1][0], s[1][1]
        self.theta = (theta0, theta1)
        self.agent_pos = np.array(s[0][:2]) 
        self.agent_orientation = s[0][2]
        self.set_deformed_maze(s[1])

    def get_state(self):
        return (self.agent_pos[0],self.agent_pos[1], self.agent_orientation), self.theta

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
    
    def update_belief(self):
        """"
        perform update over theta
        
        $$b'_{x,a,o}(theta) = \eta \cdot p(o|x,theta) \cdot b(theta)$$
        
        """

        new_belief = torch.zeros_like(self.belief)
        observation = self.get_observation()
        pos = (self.agent_pos[0],self.agent_pos[1],self.agent_orientation)

        for t, theta in enumerate(self.deformations):
            if self.belief[t] == 0:
                new_belief[t] = 0
                continue
            
            P_o_s_theta = np.all(self.get_observation(s = (pos,theta)) == observation) # 0 or 1 
            new_belief[t] = P_o_s_theta * self.belief[t]
        
        new_belief = new_belief / (torch.sum(new_belief) + 1e-10)

        return new_belief

    def set_rendering(self):
        self.screen_width = 800
        self.screen_height = 600
        pygame.init()  # Initialize all pygame modules
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption("Maze Environment")
        
        # Handle key events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return
            elif event.type == pygame.KEYDOWN:
                # Press 'r' to reset environment
                if event.key == pygame.K_r:
                    self.reset()
                # Press 'q' to quit
                elif event.key == pygame.K_q:
                    pygame.quit()
                    return
                # Press 's' to save current state
                elif event.key == pygame.K_s:
                    self.save_state()
                # Press space to pause/resume
                elif event.key == pygame.K_SPACE:
                    self.pause()
                # Press arrow keys for manual control
                elif event.key == pygame.K_LEFT:
                    self.step(3)  # Left action
                elif event.key == pygame.K_RIGHT:
                    self.step(1)  # Right action
                elif event.key == pygame.K_UP:
                    self.step(0)  # Up action
                elif event.key == pygame.K_DOWN:
                    self.step(2)  # Down action

        # Update display
        pygame.display.flip()

    def render(self):
        """Render the maze using Pygame"""
        
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

        # Add text for controls
        font = pygame.font.Font(None, 36)
        controls = [
            "Controls:",
            "R - Reset",
            "Q - Quit",
            "Space - Pause/Resume",
            "Arrows - Move agent"
        ]
        
        for i, text in enumerate(controls):
            text_surface = font.render(text, True, (0, 0, 0))
            self.screen.blit(text_surface, (self.screen_width - 200, 20 + i * 30))

        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    self.reset()
                elif event.key == pygame.K_q:
                    pygame.quit()
                    return
                elif event.key == pygame.K_SPACE:
                    self.pause()
                

        # Update the display
        pygame.display.flip()

        # Capture the current frame and add it to the list of frames
        frame = pygame.surfarray.array3d(self.screen)
        self.frames.append(frame)

    def reset(self, seed=42):
        randomdeformation = random.choice(self.deformations)
        self.agent_pos = [np.random.randint(1, self.max_shape[0]-1), np.random.randint(1, self.max_shape[1]-1)]
        self.agent_orientation = random.choice(self.orientations)
        self.set_deformed_maze(randomdeformation)
        self.goal_pos = self.original_maze.shape * np.array([randomdeformation[1],randomdeformation[0]])
        self.theta = randomdeformation
        self.timestep = 0
        
        self.belief = torch.ones(len(self.deformations)) / len(self.deformations)
        obs = OrderedDict({
                            "x": torch.tensor([self.agent_pos[0]],dtype=torch.int32),              # Values from 0 to 10
                            "y": torch.tensor([self.agent_pos[1]],dtype=torch.int32),              # Values from 0 to 10
                            "phi": torch.tensor([self.agent_orientation],dtype=torch.int32),             # Values from 0 to 4
                            "belief": self.belief , # Probability vector
                        })


        return obs, {}
    
class MDPGYMGridEnvDeform(gym.Env):
    
    def __init__(self, maze, l0,h0,l1,h1,render_mode = None):

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
        self.state_dict = {state : i for i, state in enumerate(self.states)}
        self.positions = [(x,y,phi) for x in range(1,self.max_shape[0]-1) for y in range(1,self.max_shape[1]-1) for phi in range(4)]
        
        self.l0 = l0
        self.h0 = h0
        self.l1 = l1
        self.h1 = h1

        self.goal_pos = self.original_maze.shape - np.array([2,2])
        
        self.frames = []
        self.reset()

        self.render_mode = render_mode
        if self.render_mode == "rgb_array":
            self.set_rendering()

        # gym attributes
        self.action_space = gym.spaces.Discrete(4)
        self.observation_space =  Dict({
                                    "x": Discrete(self.max_shape[0]),              # Values from 0 to 10
                                    "y": Discrete(self.max_shape[1]),              # Values from 0 to 10
                                    "phi": Discrete(5),                             # Values from 0 to 4
                                    "theta": Box(low=min(l0,l1), high=max(h0,h1), shape=(2,), dtype=int)  # Probability vector
                        # Probability vector
                                })
    
    def step(self, a):

        """take action a from state s (if given) or from actual state of the maze 
        
        return the next state, the reward, if the episode is terminated, if the episode is truncated, info"""
        
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
            self.agent_pos = new_pos

        phi_ = (phi + a) % 4
        
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
                
        self.timestep += 1
        truncated = self.timestep >= 100 

        if self.render_mode == "rgb_array":
            self.render()

        obs = OrderedDict({
                            "x": torch.tensor([x_],dtype=torch.int32),              # Values from 0 to 10
                            "y": torch.tensor([y_],dtype=torch.int32),              # Values from 0 to 10
                            "phi": torch.tensor([phi_],dtype=torch.int32),             # Values from 0 to 4
                            "theta": torch.tensor(self.theta) , # Probability vector
                        })

        
        return obs, reward, terminated, truncated, info
    
    def set_render_mode(self, mode):
        self.render_mode = mode
        if self.render_mode == "human":
            self.set_rendering()
        
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

    def set_state(self, s):
        theta0, theta1 = s[1][0], s[1][1]
        self.theta = (theta0, theta1)
        self.agent_pos = np.array(s[0][:2]) 
        self.agent_orientation = s[0][2]
        self.set_deformed_maze(s[1])

    def get_state(self):
        return (self.agent_pos[0],self.agent_pos[1], self.agent_orientation), self.theta

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
    
    def set_rendering(self):
        self.screen_width = 800
        self.screen_height = 600
        pygame.init()  # Initialize all pygame modules
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption("Maze Environment")
        
        # Handle key events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return
            elif event.type == pygame.KEYDOWN:
                # Press 'r' to reset environment
                if event.key == pygame.K_r:
                    self.reset()
                # Press 'q' to quit
                elif event.key == pygame.K_q:
                    pygame.quit()
                    self.render_mode = None
                    return
                # Press 's' to save current state
                elif event.key == pygame.K_s:
                    self.save_state()
                # Press space to pause/resume
                elif event.key == pygame.K_SPACE:
                    self.pause()
                # Press arrow keys for manual control
                elif event.key == pygame.K_LEFT:
                    self.step(3)  # Left action
                elif event.key == pygame.K_RIGHT:
                    self.step(1)  # Right action
                elif event.key == pygame.K_UP:
                    self.step(0)  # Up action
                elif event.key == pygame.K_DOWN:
                    self.step(2)  # Down action

        # Update display
        pygame.display.flip()

    def render(self):
        """Render the maze using Pygame"""
        
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

        # Add text for controls
        font = pygame.font.Font(None, 36)
        controls = [
            "Controls:",
            "R - Reset",
            "Q - Quit",
            "Space - Pause/Resume",
            "Arrows - Move agent"
        ]
        
        for i, text in enumerate(controls):
            text_surface = font.render(text, True, (0, 0, 0))
            self.screen.blit(text_surface, (self.screen_width - 200, 20 + i * 30))

        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    self.reset()
                elif event.key == pygame.K_q:
                    pygame.quit()
                    return
                elif event.key == pygame.K_SPACE:
                    self.pause()
                elif event.key == pygame.K_LEFT:
                    self.step(3)
                elif event.key == pygame.K_RIGHT:
                    self.step(1)
                elif event.key == pygame.K_UP:
                    self.step(0)
                elif event.key == pygame.K_DOWN:
                    self.step(2)
                

        # Update the display
        pygame.display.flip()

        # Capture the current frame and add it to the list of frames
        frame = pygame.surfarray.array3d(self.screen)
        self.frames.append(frame)

    def reset(self,seed=42,state=None): 
        """
        Reset the environment to the initial state.
        optional: set the state to a specific state 
            s = (x,y,phi,theta) or s = (x,y,phi) or s = (theta)
        """

        if isinstance(state, tuple) and isinstance(state[0], tuple): # pos and theta
            self.set_state(state)
        elif isinstance(state, tuple) and isinstance(state[0], int) and len(state) == 3: # pos
            raise NotImplementedError("passing only state x y phi not implemented yet")
        elif isinstance(state, tuple) and isinstance(state[0], int) and len(state) == 2: # theta
            #raise NotImplementedError("passing only state theta not implemented yet")
            randomdeformation = state
            self.agent_pos = [np.random.randint(1, self.max_shape[0]-1), np.random.randint(1, self.max_shape[1]-1)]
            self.agent_orientation = random.choice(self.orientations)
            self.set_deformed_maze(randomdeformation)
            self.goal_pos = self.original_maze.shape * np.array([randomdeformation[1],randomdeformation[0]])
            self.theta = randomdeformation

        elif state is None:
            randomdeformation = random.choice(self.deformations)
            self.agent_pos = [np.random.randint(1, self.max_shape[0]-1), np.random.randint(1, self.max_shape[1]-1)]
            self.agent_orientation = random.choice(self.orientations)
            self.set_deformed_maze(randomdeformation)
            self.goal_pos = self.original_maze.shape * np.array([randomdeformation[1],randomdeformation[0]])
            self.theta = randomdeformation
        
        self.timestep = 0
        
        self.belief = torch.ones(len(self.deformations)) / len(self.deformations)

        obs = OrderedDict({
                    "x": torch.tensor([self.agent_pos[0]],dtype=torch.int32),              # Values from 0 to 10
                    "y": torch.tensor([self.agent_pos[1]],dtype=torch.int32),              # Values from 0 to 10
                    "phi": torch.tensor([self.agent_orientation],dtype=torch.int32),             # Values from 0 to 4
                    "theta": torch.tensor(self.theta) , # Probability vector
                })

        if self.render_mode == "rgb_array":
            self.render()

        return obs, {}           

###----------------------------CONTINOUS----------------------------###

DEFAULT_OBSTACLES = [((0.14625, 0.3325), (0.565, 0.55625)), 
             ((0.52875, 0.5375), (0.7375, 0.84125)), 
             ((0.0, 0.00125), (0.01625, 0.99125)), 
             ((0.0075, 0.00125), (0.99875, 0.04)), 
             ((0.98875, 0.0075), (0.99875, 1.0)), 
             ((0.00125, 0.9825), (0.99875, 1.0))]

class ObservableDeformedGridworld(gym.Env):

    def __init__(self, grid_size=(1.0, 1.0), step_size=0.02, goal=(0.9, 0.9), 
                 obstacles=DEFAULT_OBSTACLES, stretch=(1.0, 1.0), shear=(0.0, 0.0), observation_radius=0.05, render_mode=None,shear_range=(-0.2,0.2),stretch_range=(0.4,1)):
        """
        Initialize the observable deformed continuous gridworld.
        :param grid_size: Size of the grid (width, height).
        :param step_size: Step size for the agent's movement.
        :param goal: Coordinates of the goal position.
        :param obstacles: List of obstacles as rectangles [(x_min, y_min), (x_max, y_max)].
        :param stretch: Tuple (s_x, s_y) for stretching the grid in x and y directions.
        :param shear: Tuple (sh_x, sh_y) for shearing the grid.
        :param observation_radius: Radius within which the agent can observe its surroundings.
        """
        self.grid_size = np.array(grid_size)
        self.step_size = step_size
        self.goal = np.array(goal)
        self.state = np.array([0.1, 0.1])  # Start at the origin
        self.obstacles = obstacles if obstacles else []
        self.observation_radius = observation_radius

        # Transformation matrix
        self.transformation_matrix = np.array([
            [stretch[0], shear[0]],
            [shear[1], stretch[1]]
        ])
        self.inverse_transformation_matrix = np.linalg.inv(self.transformation_matrix)

        # Rendering mode
        self.render_mode = render_mode

        # gymnasium compatibility
        self.action_space = gym.spaces.Discrete(4)
        self.observation_space =  Dict({
            "pos": gym.spaces.Box(low=1.0, high=1.0, shape=(2,),dtype=float),
            "theta": gym.spaces.Box(low=-1.0, high=1.0, shape=(4,),dtype=float), # deformation is a 2x2 tensor
        })

        self.stretch_range = stretch_range
        self.shear_range = shear_range

        self.timestep = 0

        self.corners = [
            np.array([0, 0]),
            np.array([self.grid_size[0], 0]),
            self.grid_size,
            np.array([0, self.grid_size[1]]),
        ]
        self.transformed_corners = [self.transform(corner) for corner in self.corners]
        
    def reset(self,seed=None):
        """
        Reset the environment to the initial state.
        :return: Initial state and observation.
        """
        np.random.seed(seed)
        self.set_deformation(self.sample(2,self.stretch_range), self.sample(2,self.shear_range))  # Reset deformation to random
        self.transformed_corners = [self.transform(corner) for corner in self.corners]
        # self.state = np.array([0.1, 0.1])  # Start at the origin
        #self.state = np.random.rand(2) * self.transform(self.grid_size) # Random start position in the deformable grid
        self.state = sample_in_parallelogram(self.transformed_corners)


        state = OrderedDict({
            "pos": self.state,
            "theta": self.transformation_matrix.flatten(),
        }) 
        
        self.timestep = 0

        # print(f"Initial agent position: {self.state}",
        #       f"Initial goal position: {self.goal}",
        #       f"Initial deformation: {self.transformation_matrix}",
        #       f"Initial observation: {self.observe_obstacle()}",
        #       sep="\n")
        
        return state, {}
    
    def set_deformation(self, stretch, shear):
        """
        Set the deformation transformation matrix based on stretch and shear parameters.
        
        This function creates a transformation matrix to apply grid deformations, including 
        stretching and shearing, to the grid coordinates. It also computes the inverse of 
        this transformation for reversing the deformation.

        :param stretch: A tuple (s_x, s_y) for stretching the grid in the x and y directions.
        :param shear: A tuple (sh_x, sh_y) for shearing the grid in the x and y directions.
        """
        # Create the transformation matrix based on stretch and shear
        self.transformation_matrix = np.array([
            [stretch[0], shear[0]],  # First row: stretch in x and shear in x direction
            [shear[1], stretch[1]]   # Second row: shear in y and stretch in y direction
        ])

        # Calculate the inverse transformation matrix for reversing the deformation
        self.inverse_transformation_matrix = np.linalg.inv(self.transformation_matrix)

        # Optionally, print the transformation matrices for debugging
        # print(f"Transformation Matrix:\n{self.transformation_matrix}")
        # print(f"Inverse Transformation Matrix:\n{self.inverse_transformation_matrix}")

    def set_pos(self, pos):
        """
        Set the agent's state to a new position.
        
        This function directly updates the agent's position (state) to the provided coordinates.

        :param pos: A tuple or array representing the new position of the agent in the grid.
        """
        # Update the state (agent's position)
        self.state = np.array(pos)

        # Optionally, print the new state for debugging
        # print(f"New agent position: {self.state}")

    def transform(self, position):
        """
        Apply the grid deformation to a given position.
        :param position: (x, y) in original space.
        :return: Transformed position in the deformed grid.
        """
        return np.dot(self.transformation_matrix, position)

    def inverse_transform(self, position):
        """
        Map a position from the deformed grid back to the original space.
        :param position: (x, y) in the deformed grid.
        :return: Original position.
        """
        return np.dot(self.inverse_transformation_matrix, position)
    
    def is_in_obstacle(self, position):
        """
        Check if a given position is inside any obstacle.
        :param position: The (x, y) coordinates to check in the original space.
        :return: True if the position is inside an obstacle, False otherwise.
        """
        for obs in self.obstacles:
            (x_min, y_min), (x_max, y_max) = obs
            bottom_left = self.transform(np.array([x_min, y_min]))
            bottom_right = self.transform(np.array([x_max, y_min]))
            top_left = self.transform(np.array([x_min, y_max]))
            top_right = self.transform(np.array([x_max, y_max]))
            obstacle = [bottom_left, bottom_right, top_right, top_left]
            if is_point_in_parallelogram(position, obstacle):
                return True
        return False

    def observe_single_obstacle(self):
        """
        Check if any part of an obstacle is within the observation radius of the agent.
        :return: True if any part of an obstacle is within the observation radius, False otherwise.
        """
        for obs in self.obstacles:
            (x_min, y_min), (x_max, y_max) = obs
            
            # Clamp the agent's position to the obstacle's boundaries to find the closest point
            closest_x = np.clip(self.state[0], x_min, x_max)
            closest_y = np.clip(self.state[1], y_min, y_max)
            
            # Compute the distance from the agent to this closest point
            closest_point = np.array([closest_x, closest_y])
            distance_to_obstacle = np.linalg.norm(self.state - closest_point)
            
            # Check if this distance is within the observation radius
            if distance_to_obstacle <= self.observation_radius:
                return 1
        
        return 0
    
    def observe_obstacle(self):
        """
        Efficiently and precisely check for obstacles in the four cardinal directions (N, E, S, W).
        Each direction checks for obstacles in a quarter-circle arc within the observation radius.
        :return: A numpy array of shape (4,), where each entry indicates the presence of obstacles 
                in the respective direction (North, East, South, West).
        """
        directions = ["N", "E", "S", "W"]
        obstacle_presence = np.zeros(4)  # Default: no obstacles in any direction

        # Precompute direction boundaries in radians
        direction_ranges = [
            (315, 45),   # North: [-45°, +45°]
            (45, 135),   # East: [+45°, +135°]
            (135, 225),  # South: [+135°, +225°]
            (225, 315)   # West: [+225°, +315°]
        ]
        direction_ranges_rad = [(np.deg2rad(a1), np.deg2rad(a2)) for a1, a2 in direction_ranges]

        for obs in self.obstacles:
            (x_min, y_min), (x_max, y_max) = obs
            x_min, y_min = self.transform([x_min, y_min])
            x_max, y_max = self.transform([x_max, y_max])

            # Generate sampled points along the edges of the obstacle
            num_samples = 5  # Increase for more precision
            edge_points = np.concatenate([
                np.linspace([x_min, y_min], [x_max, y_min], num_samples),  # Bottom edge
                np.linspace([x_max, y_min], [x_max, y_max], num_samples),  # Right edge
                np.linspace([x_max, y_max], [x_min, y_max], num_samples),  # Top edge
                np.linspace([x_min, y_max], [x_min, y_min], num_samples)   # Left edge
            ])

            # Compute vectors from agent to sampled points
            vectors = edge_points - self.state
            distances = np.linalg.norm(vectors, axis=1)

            # Filter points that are outside the observation radius
            within_radius = distances <= self.observation_radius
            if not np.any(within_radius):
                continue  # Skip obstacles entirely outside the radius

            # Compute angles relative to positive Y-axis
            angles = np.arctan2(vectors[:, 1], vectors[:, 0])  # Radians
            angles = (angles + 2 * np.pi) % (2 * np.pi)  # Normalize to [0, 2π)

            # Check which direction each point falls into
            for i, (angle_min, angle_max) in enumerate(direction_ranges_rad):
                if obstacle_presence[i] == 1:
                    continue  # Early exit if the direction is already flagged
                for angle in angles[within_radius]:
                    if (angle_min <= angle < angle_max) or (
                        angle_max < angle_min and (angle >= angle_min or angle < angle_max)
                    ):
                        obstacle_presence[i] = 1
                        break  # No need to check further points for this direction

        return obstacle_presence
        
    def step(self, action):
        """
        Take a step in the environment, interpreting the action in the deformed space.
        :param action: One of ['N', 'S', 'E', 'W'].
        :return: Tuple (next_state, observation, reward, done, info).
        """
        # Map actions to movements in the deformed space
        moves = [np.array([0, self.step_size]),   # Move up in deformed space
            np.array([0, -self.step_size]),  # Move down in deformed space
            np.array([self.step_size, 0]),   # Move right in deformed space
            np.array([-self.step_size, 0])   # Move left in deformed space
        ]

        # Get the movement vector in the deformed space
        move = moves[action]

        # Map the movement to the original space using the inverse transformation
        # move_original = np.dot(self.inverse_transformation_matrix, move)

        # Update state in the original grid space
        next_state = self.state + move

        num_samples = 10  # Number of points to sample along the path
        path = np.linspace(self.state, next_state, num_samples)

        # Check for collisions along the path
        collision = any(self.is_in_obstacle(point) for point in path)

        if np.linalg.norm(next_state - self.transform(self.goal)) < self.observation_radius:
            terminated = True
            reward = 1.0 
            info = {"collision": False, "out": False, 'goal': True}
        # Check if the is inside the deformed grid boundaries
        elif not is_point_in_parallelogram(next_state, self.transformed_corners):
            reward = -2.0
            info = {"out": True}
            next_state = self.state
            terminated = False
        # Check if the new state is in an obstacle
        elif collision:   
            reward = -2.0  # Penalty for hitting an obstacle
            info = {"collision": True}
            terminated = False
        else:
            terminated = False
            reward = -0.5
            info = {"collision": False, "out": False, "goal": False}
    
        self.state = next_state
        self.timestep += 1
        truncated = self.timestep > 500 

        if self.render_mode == "human":
            self.render()

        state = OrderedDict({
                    "pos": self.state,
                    "theta": self.transformation_matrix.flatten(),
                })

        # Return the transformed state, reward, and terminated truncated flag
        return state, reward, terminated, truncated, info
    
    def close(self):
        self.render_mode = None
        pygame.quit()
    
    def sample(self,num,limit):
        low,high = limit
        return low + np.random.rand(num)*(high-low)
    
    def render(self):
        """
        Render the deformed gridworld environment along with the original gridworld.
        The original gridworld serves as a reference background.
        """
        import pygame  # Ensure Pygame is imported

        # Define colors
        WHITE = (255, 255, 255)
        LIGHT_GRAY = (200, 200, 200)
        BLUE = (0, 0, 255)
        GREEN = (0, 255, 0)
        RED = (255, 0, 0)
        PINK = (255, 105, 180)  
        YELLOW = (255, 255, 0)
        BLACK = (0, 0, 0)

        # Initialize the screen
        if not hasattr(self, "screen"):
            self.screen_width = 1000
            self.screen_height = 1000
            pygame.init()
            self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
            pygame.display.set_caption("Deformed and Original Gridworld")

        # Fill background with white
        self.screen.fill(WHITE)

    # Compute the bounding box of the deformed grid
        corners = [
            np.array([0, 0]),
            np.array([self.grid_size[0], 0]),
            self.grid_size,
            np.array([0, self.grid_size[1]]),
        ]
        transformed_corners = [self.transform(corner) for corner in corners]
        x_coords, y_coords = zip(*transformed_corners)
        min_x, max_x = min(x_coords), max(x_coords)
        min_y, max_y = min(y_coords), max(y_coords)

        # Define scaling factors to fit the deformed grid within the screen
        scale_x = self.screen_width / (max_x - min_x)
        scale_y = self.screen_height / (max_y - min_y)
        scale = min(scale_x, scale_y)  # Uniform scaling to maintain aspect ratio

        # Add upward translation offset
        y_translation = max(0, -min_y * scale)

        # Transform helper for rendering
        def to_screen_coords(pos):
            """
            Map transformed coordinates to screen coordinates, scaled and shifted to fit the screen.
            """
            x, y = pos
            x_screen = int((x - min_x) * scale)
            y_screen = int((max_y - y) * scale + y_translation)  # Flip y-axis and add upward translation
            return x_screen, y_screen
        
        # Draw the un-deformed grid (background)
        for i in range(int(self.grid_size[0]) + 1):
            pygame.draw.line(self.screen, LIGHT_GRAY,
                            to_screen_coords((i, 0)),
                            to_screen_coords((i, self.grid_size[1])), width=1)
        for j in range(int(self.grid_size[1]) + 1):
            pygame.draw.line(self.screen, LIGHT_GRAY,
                            to_screen_coords((0, j)),
                            to_screen_coords((self.grid_size[0], j)), width=1)

        # Draw the deformed grid boundaries
        corners = [
            np.array([0, 0]),
            np.array([self.grid_size[0], 0]),
            self.grid_size,
            np.array([0, self.grid_size[1]]),
        ]
        transformed_corners = [self.transform(corner) for corner in corners]
        pygame.draw.polygon(self.screen, BLACK, [to_screen_coords(corner) for corner in transformed_corners], width=3)

        # Draw the obstacles in both grids
        for obs in self.obstacles:
            (x_min, y_min), (x_max, y_max) = obs
            # Original obstacle
            pygame.draw.rect(self.screen, PINK,
                            (*to_screen_coords((x_min, y_max)),  # Top-left corner
                            int((x_max - x_min) * scale),      # Width
                            int((y_max - y_min) * scale)),    # Height
                            width=0)

            # Transformed obstacle
            bottom_left = self.transform(np.array([x_min, y_min]))
            bottom_right = self.transform(np.array([x_max, y_min]))
            top_left = self.transform(np.array([x_min, y_max]))
            top_right = self.transform(np.array([x_max, y_max]))
            pygame.draw.polygon(self.screen, RED, [
                to_screen_coords(bottom_left),
                to_screen_coords(bottom_right),
                to_screen_coords(top_right),
                to_screen_coords(top_left)
            ])

        # Draw the agent in both grids
        agent_position = self.state
        transformed_agent_position = agent_position
        pygame.draw.circle(self.screen, BLUE, to_screen_coords(agent_position), 10)  # Original
        pygame.draw.circle(self.screen, GREEN, to_screen_coords(transformed_agent_position), 10)  # Transformed

        # Draw the goal in both grids
        goal_position = self.goal
        transformed_goal_position = self.transform(goal_position)
        pygame.draw.circle(self.screen, GREEN, to_screen_coords(goal_position), 12)  # Original
        pygame.draw.circle(self.screen, YELLOW, to_screen_coords(transformed_goal_position), 12)  # Transformed

        # Draw observation radius as a dashed circle around the agent
        observation_radius = self.observation_radius # stays the same in both grids
        pygame.draw.circle(self.screen, YELLOW, to_screen_coords(agent_position), 
                        int(self.observation_radius * scale), 1)  # Original
        pygame.draw.circle(self.screen, YELLOW, to_screen_coords(transformed_agent_position), 
                        int(observation_radius * scale), 1)  # Transformed

        # Update the display
        pygame.display.flip()

        # Handle key events
        # Handle key events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return
            elif event.type == pygame.KEYDOWN:
                # Press 'r' to reset environment
                if event.key == pygame.K_r:
                    self.reset()
                # Press 'w' to quit
                elif event.key == pygame.K_w:
                    pygame.quit()
                    return
                # Press 's' to save current state
                elif event.key == pygame.K_s:
                    self.save_state()
                # Press space to pause/resume
                elif event.key == pygame.K_SPACE:
                    self.pause()
                # Press arrow keys for manual control
                elif event.key == pygame.K_LEFT:
                    return self.step(3)  # Left action
                elif event.key == pygame.K_RIGHT:
                    return self.step(2)  # Right action
                elif event.key == pygame.K_UP:
                    return self.step(0)  # Up action
                elif event.key == pygame.K_DOWN:
                    return self.step(1)  # Down action
        return None, None, None, None, None

    def set_pos_nodeform(self):
        """
        Set the agent's state to a new position.
        
        This function directly updates the agent's position (state) to the provided coordinates.

        :param pos: A tuple or array representing the new position of the agent in the grid.
        """
        low, high = -.2, 1.2 # depends on the shear (nto stretch since compression)
        pos = self.sample(2,limit=(low,high))
        
        # Update the state (agent's position)
        self.state = np.array(pos)

        # Optionally, print the new state for debugging
        # print(f"New agent position: {self.state}")
        return pos

    def is_in_obstacle_nodeform(self, position):
        """
        Check if a given position is inside any obstacle.
        :param position: The (x, y) coordinates to check in the original space.
        :return: True if the position is inside an obstacle, False otherwise.
        """
        for obs in self.obstacles:
            (x_min, y_min), (x_max, y_max) = obs
            if x_min <= position[0] <= x_max and y_min <= position[1] <= y_max:            
                return True
        return False

class Grid(gridworld.ObservableDeformedGridworld,gym.Env):
   
    def __init__(self, grid_size=(1.0, 1.0), step_size=0.02, goal=(0.9, 0.9), obstacles=DEFAULT_OBSTACLES,
                  stretch=(1.0, 1.0), shear=(0.0, 0.0), observation_radius=0.05, shear_range=(-0.2,0.2), 
                  stretch_range=(0.4,1), render_mode=None,max_timesteps=500):
        super().__init__(grid_size, step_size, goal, obstacles, stretch, shear, observation_radius, shear_range, stretch_range)
        self.render_mode = render_mode

        self.action_space = gym.spaces.Discrete(4)
        self.observation_space =  gym.spaces.Dict({
            "pos": gym.spaces.Box(low=.0, high=1.0, shape=(2,),dtype=float),
            "theta": gym.spaces.Box(low=.0, high=1.0, shape=(4,),dtype=float), # deformation is a 2x2 tensor
        })    
    
    def render(self):
        """
        Render the deformed gridworld environment along with the original gridworld.
        The original gridworld serves as a reference background.
        """
        import pygame  # Ensure Pygame is imported

        # Define colors
        WHITE = (255, 255, 255)
        LIGHT_GRAY = (200, 200, 200)
        BLUE = (0, 0, 255)
        GREEN = (0, 255, 0)
        RED = (255, 0, 0)
        PINK = (255, 105, 180)  
        YELLOW = (255, 255, 0)
        BLACK = (0, 0, 0)

        # Initialize the screen
        if not hasattr(self, "screen"):
            self.screen_width = 1000
            self.screen_height = 1000
            pygame.init()
            self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
            pygame.display.set_caption("Deformed and Original Gridworld")

        # Fill background with white
        self.screen.fill(WHITE)

    # Compute the bounding box of the deformed grid
        corners = [
            np.array([0, 0]),
            np.array([self.grid_size[0], 0]),
            self.grid_size,
            np.array([0, self.grid_size[1]]),
        ]
        transformed_corners = [self.transform(corner) for corner in corners]
        x_coords, y_coords = zip(*transformed_corners)
        min_x, max_x = min(x_coords), max(x_coords)
        min_y, max_y = min(y_coords), max(y_coords)

        # Define scaling factors to fit the deformed grid within the screen
        scale_x = self.screen_width / (max_x - min_x)
        scale_y = self.screen_height / (max_y - min_y)
        scale = min(scale_x, scale_y)  # Uniform scaling to maintain aspect ratio

        # Add upward translation offset
        y_translation = max(0, -min_y * scale)

        # Transform helper for rendering
        def to_screen_coords(pos):
            """
            Map transformed coordinates to screen coordinates, scaled and shifted to fit the screen.
            """
            x, y = pos
            x_screen = int((x - min_x) * scale)
            y_screen = int((max_y - y) * scale + y_translation)  # Flip y-axis and add upward translation
            return x_screen, y_screen
        
        # Draw the un-deformed grid (background)
        for i in range(int(self.grid_size[0]) + 1):
            pygame.draw.line(self.screen, LIGHT_GRAY,
                            to_screen_coords((i, 0)),
                            to_screen_coords((i, self.grid_size[1])), width=1)
        for j in range(int(self.grid_size[1]) + 1):
            pygame.draw.line(self.screen, LIGHT_GRAY,
                            to_screen_coords((0, j)),
                            to_screen_coords((self.grid_size[0], j)), width=1)

        # Draw the deformed grid boundaries
        corners = [
            np.array([0, 0]),
            np.array([self.grid_size[0], 0]),
            self.grid_size,
            np.array([0, self.grid_size[1]]),
        ]
        transformed_corners = [self.transform(corner) for corner in corners]
        pygame.draw.polygon(self.screen, BLACK, [to_screen_coords(corner) for corner in transformed_corners], width=3)

        # Draw the obstacles in both grids
        for obs in self.obstacles:
            (x_min, y_min), (x_max, y_max) = obs
            # Original obstacle
            pygame.draw.rect(self.screen, PINK,
                            (*to_screen_coords((x_min, y_max)),  # Top-left corner
                            int((x_max - x_min) * scale),      # Width
                            int((y_max - y_min) * scale)),    # Height
                            width=0)

            # Transformed obstacle
            bottom_left = self.transform(np.array([x_min, y_min]))
            bottom_right = self.transform(np.array([x_max, y_min]))
            top_left = self.transform(np.array([x_min, y_max]))
            top_right = self.transform(np.array([x_max, y_max]))
            pygame.draw.polygon(self.screen, RED, [
                to_screen_coords(bottom_left),
                to_screen_coords(bottom_right),
                to_screen_coords(top_right),
                to_screen_coords(top_left)
            ])

        # Draw the agent in both grids
        agent_position = self.state
        transformed_agent_position = agent_position
        pygame.draw.circle(self.screen, BLUE, to_screen_coords(agent_position), 10)  # Original
        pygame.draw.circle(self.screen, GREEN, to_screen_coords(transformed_agent_position), 10)  # Transformed

        # Draw the goal in both grids
        goal_position = self.goal
        transformed_goal_position = self.transform(goal_position)
        pygame.draw.circle(self.screen, GREEN, to_screen_coords(goal_position), 12)  # Original
        pygame.draw.circle(self.screen, YELLOW, to_screen_coords(transformed_goal_position), 12)  # Transformed

        # Draw observation radius as a dashed circle around the agent
        observation_radius = self.observation_radius # stays the same in both grids
        pygame.draw.circle(self.screen, YELLOW, to_screen_coords(agent_position), 
                        int(self.observation_radius * scale), 1)  # Original
        pygame.draw.circle(self.screen, YELLOW, to_screen_coords(transformed_agent_position), 
                        int(observation_radius * scale), 1)  # Transformed

        # Update the display
        pygame.display.flip()

        # Handle key events
        # Handle key events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return
            elif event.type == pygame.KEYDOWN:
                # Press 'r' to reset environment
                if event.key == pygame.K_r:
                    self.reset()
                # Press 'w' to quit
                elif event.key == pygame.K_w:
                    pygame.quit()
                    return
                # Press 's' to save current state
                elif event.key == pygame.K_s:
                    self.save_state()
                # Press space to pause/resume
                elif event.key == pygame.K_SPACE:
                    self.pause()
                # Press arrow keys for manual control
                elif event.key == pygame.K_LEFT:
                    return self.step(3)  # Left action
                elif event.key == pygame.K_RIGHT:
                    return self.step(2)  # Right action
                elif event.key == pygame.K_UP:
                    return self.step(0)  # Up action
                elif event.key == pygame.K_DOWN:
                    return self.step(1)  # Down action
        return None, None, None, None, None
    
    def reset(self, seed=None):
        """
        Reset the environment to the initial state.
        """
        super().reset(seed=seed)
        
        state = OrderedDict({
            "pos": np.array(self.state),
            "theta": np.array(self.transformation_matrix).flatten(),
        })

        return state, {}
    
    def step(self, action):
        if self.render_mode == "human":
            self.render()

        return super().step(action)
      
    def close(self):
        """
        Close the Pygame window.
        """
        pygame.quit()

class POMDPDeformedGridworld(Grid):
    def __init__(self, render_mode='human'):
        super(POMDPDeformedGridworld, self).__init__(render_mode=render_mode)
        
        self.observation_space = Dict({
            'obs': Discrete(2),
            'pos': Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32)
        })
    
    def reset(self, seed=None):
        state, _ = super().reset(seed=seed)
        pomdp_state = {
            'obs': torch.tensor([1], dtype=torch.float32) if super().is_collision(state['pos']) else torch.tensor([0], dtype=torch.float32),
            'pos': torch.tensor(state['pos'], dtype=torch.float32)
            }
        return pomdp_state, {}
    
    def step(self, action):
        state, reward, terminated, truncated, info = super().step(action)
        pomdp_state = {
            'obs': torch.tensor([1], dtype=torch.float32) if super().is_collision(state['pos']) else torch.tensor([0], dtype=torch.float32),
            'pos': torch.tensor(state['pos'], dtype=torch.float32)
            }
        return pomdp_state, reward, terminated,truncated, info
    
    def get_state(self):
        pomdp_state = {
            'obs': torch.tensor([1], dtype=torch.float32) if super().is_collision(self.state) else torch.tensor([0], dtype=torch.float32),
            'pos': torch.tensor(self.state, dtype=torch.float32)
            }
        return pomdp_state