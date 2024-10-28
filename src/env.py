import gymnasium as gym 
import numpy as np
import matplotlib.pyplot as plt
import random
import itertools

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
                

        self.goal_pos = self.original_maze.shape - np.array([2,2])
        
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

        """take action a from state s (if given) or from actual state of the maze """
        
        if s is not None:
            self.set_deformed_maze(s[1])
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
            reward =  10            
        elif np.all((x_,y_) == (x,y)):
            # if the agent has not moved (only at the boundary of the maze)
            reward =  -100/(self.max_shape[0]*self.max_shape[1])
        elif self.maze[x_, y_] == 1:
            # if the agent has entered a wall
            reward =  -100/(self.max_shape[0]*self.max_shape[1])
        elif self.maze[x_, y_] == 0:
            # if the agent has moved to a free cell
            reward =  -1/(self.max_shape[0]*self.max_shape[1])

        info = {}
        truncated = False 

        s_ = ((x_, y_, phi_), self.theta)
        
        if execute:
            self.timestep += 1
        
        return s_, reward, terminated, truncated, info, 
    
    def get_observation(self, s):

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

    def reset(self, seed=42):
        randomdeformation = random.choice(self.deformations)
        self.agent_pos = [np.random.randint(1, self.max_shape[0]-1), np.random.randint(1, self.max_shape[1]-1)]
        self.agent_orientation = random.choice(self.orientations)
        self.set_deformed_maze(randomdeformation)
        self.goal_pos = self.original_maze.shape * np.array([randomdeformation[1],randomdeformation[0]])
        self.theta = randomdeformation
        self.timestep = 0
        return (self.agent_pos[0],self.agent_pos[1], self.agent_orientation), (self.maze.shape[0], self.maze.shape[1])
        
    def is_done(self):
        return np.all(self.agent_pos == self.goal_pos)
    
    def is_new(self):
        return self.timestep == 0
    
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