# Table of Contents
- [Table of Contents](#table-of-contents)
- [Surgical Robot Lung Exploration Simulation](#surgical-robot-lung-exploration-simulation)
  - [Introduction](#introduction)
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
The goal of this project is to simulate a surgical robot that can explore the lung of a patient to find a tumor.
The following is an abstract simulation of the robot in action.

## Simulation
The environment is a high level simulation of the real phenomena. The robot is a ball that can move in a 2D maze. The goal is to reach the red ball that represents the lesion, being able to move in a partially observed environment.  

### State structure

The observation consists of a dictionary with information about the robot's position and goal. The dictionary consists of the following 4 keys:

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
- [ ] Multimodal Architecture ?? 
- [ ] add observation of the maze in robot proximity
- [ ] Maze deformation
- [ ] POMDP

# Ideas
- https://isaac-sim.github.io/IsaacLab/source/overview/reinforcement-learning/rl_frameworks.html
- https://github.com/Denys88/rl_games
- https://deepmind.google/discover/blog/rt-2-new-model-translates-vision-and-language-into-action/