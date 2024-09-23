# Table of Contents
- [Table of Contents](#table-of-contents)
- [Surgical Robot Lung Exploration Simulation](#surgical-robot-lung-exploration-simulation)
  - [Introduction](#introduction)
  - [Simulation](#simulation)
    - [Libraries](#libraries)
    - [Action Space](#action-space)
    - [Observation Space](#observation-space)
    - [Reward Function](#reward-function)
  - [Agent and Training](#agent-and-training)
- [Run](#run)
- [TODO](#todo)

# Surgical Robot Lung Exploration Simulation

## Introduction
The goal of this project is to simulate a surgical robot that can explore the lung of a patient to find a tumor.
The following is an abstract simulation of the robot in action.

## Simulation
The environment is built upon the (2D) Maze Environments of Gymnasium-Robotics in which A 2-DoF force-controlled ball has to navigate through a maze to reach certain goal position.
### Libraries
    - Gymnasium-Robotics
    - StableBaselines3

### Action Space

The action space is a `Box(-1.0, 1.0, (2,), float32)`. An action represents the linear force exerted on the green ball in the x and y directions.
In addition, the ball velocity is clipped in a range of `[-5, 5] m/s` in order for it not to grow unbounded.

| Num | Action                          | Control Min | Control Max | Name (in corresponding XML file)| Joint | Unit     |
| --- | --------------------------------| ----------- | ----------- | --------------------------------| ----- | ---------|
| 0   | Linear force in the x direction | -1          | 1           | motor_x                         | slide | force (N)|
| 1   | Linear force in the y direction | -1          | 1           | motor_y                         | slide | force (N)|

### Observation Space


The observation consists of a dictionary with information about the robot's position and goal. The dictionary consists of the following 4 keys:

* `observation`: its value is an `ndarray` of shape `(4,)`. It consists of kinematic information of the force-actuated ball. The elements of the array correspond to the following:

    | Num | Observation                                              | Min    | Max    | Joint Name (in corresponding XML file) |Joint Type| Unit          |
    |-----|--------------------------------------------------------- |--------|--------|----------------------------------------|----------|---------------|
    | 0   | x coordinate of the green ball in the MuJoCo simulation  | -Inf   | Inf    | ball_x                                 | slide    | position (m)  |
    | 1   | y coordinate of the green ball in the MuJoCo simulation  | -Inf   | Inf    | ball_y                                 | slide    | position (m)  |
    | 2   | Green ball linear velocity in the x direction            | -Inf   | Inf    | ball_x                                 | slide    | velocity (m/s)|
    | 3   | Green ball linear velocity in the y direction            | -Inf   | Inf    | ball_y                                 | slide    | velocity (m/s)|

* `desired_goal`: this key represents the final goal to be achieved. In this environment it is a 2-dimensional `ndarray`, `(2,)`, that consists of the two cartesian coordinates of the desired final ball position `[x,y]`. The elements of the array are the following:

    | Num | Observation                                  | Min    | Max    | Site Name (in corresponding XML file) |Unit          |
    |-----|----------------------------------------------|--------|--------|---------------------------------------|--------------|
    | 0   | Final goal ball position in the x coordinate | -Inf   | Inf    | target                                | position (m) |
    | 1   | Final goal ball position in the y coordinate | -Inf   | Inf    | target                                | position (m) |

* `achieved_goal`: this key represents the current state of the green ball, as if it would have achieved a goal. This is useful for goal orientated learning algorithms such as those that use [Hindsight Experience Replay](https://arxiv.org/abs/1707.01495) (HER).
    The value is an `ndarray` with shape `(2,)`. The elements of the array are the following:

    | Num | Observation                                    | Min    | Max    | Joint Name (in corresponding XML file) |Unit         |
    |-----|------------------------------------------------|--------|--------|---------------------------------------|--------------|
    | 0   | Current goal ball position in the x coordinate | -Inf   | Inf    | ball_x                                | position (m) |
    | 1   | Current goal ball position in the y coordinate | -Inf   | Inf    | ball_y                                | position (m) |

* `maze_map`: this key represents the maze map. The value is an `ndarray` with shape `(n*m)`, The elements of the array are the following:

    | Num | Observation | Min | Max | Unit |
    |-----|-------------|-----|-----|------|
    | 0   | Maze map    | 0   | 1   | -    |

### Reward Function

The reward can be initialized as `sparse` or `dense`:
- *sparse*: the returned reward can have two values: `0` if the ball hasn't reached its final target position, and `1` if the ball is in the final target position (the ball is considered to have reached the goal if the Euclidean distance between both is lower than 0.5 m).
- *dense*: the returned reward is the negative Euclidean distance between the achieved goal position and the desired goal.

## Agent and Training

The agent is trained using PPO algorithm.
Available model chekcpoints are stored in the `models` directory.
    
    - lungs_ppo_100k
    - lungs_ppo_1M


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
- [ ] Migrate to RL-games