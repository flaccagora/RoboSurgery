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
