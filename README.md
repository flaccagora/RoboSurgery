# Table of Contents
- [Table of Contents](#table-of-contents)
- [Surgical Robot Lung Exploration Simulation](#surgical-robot-lung-exploration-simulation)
  - [Introduction](#introduction)
  - [POMDP framework](#pomdp-framework)
    - [Belief State](#belief-state)
  - [Simulation](#simulation)
    - [State structure](#state-structure)
    - [Action Space](#action-space)
    - [Reward Function](#reward-function)
  - [Agent and Training](#agent-and-training)
- [Run](#run)
- [TODO](#todo)
- [Ideas](#ideas)

# Surgical Robot Lung Exploration Simulation

## Introduction
The goal of this project is to simulate a surgical robot that can explore the lung of a patient to find a lesion, moreover the robot is going to navigate through a deformed lung and it should determine the parameters of this tranformation to gain insights about it's location.
\
The following is an abstract simulation of the robot in action.


## POMDP framework

Formally, a POMDP is a 7-tuple $(S,A,T,R,\Omega,O,\gamma)$, where

  $S$ is a set of states, \
  $A$ is a set of actions, \
  $T$ is a set of conditional transition probabilities between states, \
  $R: S\times A \rightarrow \mathbb{R}$  is the reward function. \
  $\Omega$ is a set of observations, \
  $O$ is a set of conditional observation probabilities, \
  $\gamma \in [0,1)$ is the discount factor.

in this case, 

A **state** $s$ is a 6-tuple $(x,y,t_x,t_y,M,\theta)$ where: \
    * $(x,y)$ is the robot position \
    * $(t_x,t_y)$ is the target position \
    * $M$ is the maze map \
    * $\theta$ is the parameter for the deformation function $f_{\theta}$
  
Each **action** $a$ is a 2-tuple $(\Delta x, \Delta y)$ where $(\Delta x, \Delta y)$ is the movement of the robot in the x and y axis respectively. there are 4 possible actions: \
    * $(0,-1)$ move up \
    * $(1,0)$ move right \
    * $(0,1)$ move down \
    * $(-1,0)$ move left

The **conditional transition probabilities** $T(s' | s, a)$ are deterministic, the robot moves to the new state $s'$ with probability 1. if and only if 

$$s = (x,y,t_x,t_y,M,\theta)$$ 

and 

$$s'=(x',y',t_x,t_y,M,\theta)$$ 

where

$$(x',y') = (x,y) \text { or }(x',y') = (x,y) + a $$

The **reward function** $R(s,a,s')$ is defined as follows:


 $$R(s,a,s') = 
    \begin{cases}
    \frac{-0.1}{mapsize} &   s' \neq s_{goal} \wedge \text{moved} \\ 
    \frac{-0.2}{mapsize} &   s' \neq s_{goal}  \wedge \text{hit wall}\\
    1 &   s' = s_{goal} \\
    \end{cases}    
$$

The **observation space** $\Omega$ is a set of observations, in this case, the observation is a 4-tuple $(o_0,o_1,o_2,o_3)$ where $o_i$ is a boolean value that represents if there is a wall in the relative adjacent cell (up right down left).



The **conditional observation probabilities** $O(o|s,a)$ are also deterministic.


$$O(o|s,a)= O(o|s) = 
\begin{cases}
    1 &   \text{if } (x,y) \text{ adjacent cells for map } f_\theta(M) \text{are compatible with } o \\
    0 &   \text{otherwise} \\
\end{cases}
$$

Where $f$ is the deformation function that acts on the original maze map M with parameter $\theta$.

### Belief State

Because the agent does not directly observe the environment's state, the agent must make decisions under uncertainty of the true environment state. The belief function is a probability distribution over the states of the environment.

$$b : S \rightarrow [0,1] \text{ and } \sum_s b(s) = 1  $$

By interacting with the environment and receiving observations, the agent may update its belief in the true state by updating the probability distribution of the current state

$$ b'(s')=\eta O(o\mid s',a)\sum _{s\in S}T(s'\mid s,a)b(s)$$

where $η = \frac{1}{Pr ( o ∣ b , a )}$ is a normalizing constant with 
$$Pr ( o ∣ b , a ) = \sum_{s'\in S} O ( o ∣ s' , a ) \sum_{s \in S}( s'∣s,a)b(s)$$

## Simulation

The environment is a high level simulation of the real phenomena. The robot is a ball that can move in a 2D maze. The goal is to reach the red ball that represents the lesion, being able to move in a partially observed environment.  

### State structure

A state is completely specified by the following dictionary:

* `robot`: its value is an `ndarray` of shape `(2,)`. The elements of the array correspond to the following:

    | Num | Observation         |
    |-----|-------------------- |
    | 0   | Robot x coordinate  |
    | 1   | Robot y coordinate  |


* `target`: its value is an `ndarray` of shape `(2,)`. The elements of the array correspond to the following:

    | Num | Observation                                  |
    |-----|----------------------------------------------|
    | 0   | Target x coordinate position in the  |
    | 1   | Target y coordinate position in the  |


* `maze_map`: this key represents the *deformation parameter* for the maze map. The value is an `ndarray` with shape `(,)`, The elements of the array are the following:

    | Num | Observation |
    |-----|-------------|
    | 0   | x stretch    |
    | 0   | y stretch   |


### Action Space

The action space is a `Box(-1.0, 1.0, (2,), float32)`. An action represents the movement to an adjacent cell.

| Num | Action                          |  
| --- | --------------------------------|
| 0   | + (0,-1)                        |  
| 1   | + (1,0)                         |  
| 2   | + (0,1)                         |  
| 3   | + (-1,0)                        |  


### Reward Function

The reward function is defined as follows:

$$r(s, a, s') = $$



In this case, the belief state is a probability distribution over the states $s \in S$

$$b(s) = p_1(x,y)\cdot p_2(\underbar{$\theta$})\cdot p_3(t_x,t_y)$$




## Agent and Training


# Run

create conda environment
```bash
conda create -n RoboSurg python=3.9
conda activate RoboSurg
```
install requirements
```bash
pip install -r requirements.txt
```
install the package as editable to register the environment
```bash
pip install -e .
```

open and run jupyter notebook [test.ipynb](./test.ipynb)

# TODO
- [ ] add observation of the maze in robot proximity
- [ ] Maze deformation
- [ ] POMDP

# Ideas
- https://isaac-sim.github.io/IsaacLab/source/overview/reinforcement-learning/rl_frameworks.html
- https://github.com/Denys88/rl_games
- https://deepmind.google/discover/blog/rt-2-new-model-translates-vision-and-language-into-action/