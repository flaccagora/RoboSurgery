import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pygame
import time
import gymnasium as gym
from gymnasium.spaces import Dict, Discrete, Box
from collections import OrderedDict
from environment.env import ObservableDeformedGridworld


from stable_baselines3 import DQN
import numpy as np

obstacles = [((0.14625, 0.3325), (0.565, 0.55625)), 
             ((0.52875, 0.5375), (0.7375, 0.84125)), 
             ((0.0, 0.00125), (0.01625, 0.99125)), 
             ((0.0075, 0.00125), (0.99875, 0.04)), 
             ((0.98875, 0.0075), (0.99875, 1.0)), 
             ((0.00125, 0.9825), (0.99875, 1.0))]


def train_dqn(args):
    from stable_baselines3.common.callbacks import CheckpointCallback
    from wandb.integration.sb3 import WandbCallback
    import wandb

    total_timesteps = args.total_timesteps
    batch_size = args.batch_size
    lr = args.learning_rate
    target_update = args.target_update
    gamma = args.gamma
    render_mode = args.render_mode

    config = {
        "policy_type": "MultiInputPolicy",
        "env_name": "ObservableDeformedGridworld",
        "total_timesteps": total_timesteps,
        "Batch_Size": batch_size,
        'grid_size': (1.0,1.0),
        'step_size': 0.1,
        'obstacles':obstacles,
        'observation_radius':0.2,
        "shear range":None,
        "stretch range":None
    }
    run = wandb.init(
        project="DQNsb3 - MDP - ObservableDeformedGridworld",
        config=config,
        sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
        monitor_gym=True,  # auto-upload the videos of agents playing the game
        save_code=True,  # optional
    )


    # Save a checkpoint every 10000 steps
    checkpoint_callback = CheckpointCallback(
                            save_freq=500000,
                            save_path=f"DQN_continous_{run.id}",
                            name_prefix="rl_model",
                            save_replay_buffer=False,
                            save_vecnormalize=True,
                        )

    callbacks = [ WandbCallback(
                                verbose=2,
                                log="parameters",
                                ),
                checkpoint_callback,
                ]


    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder

    def make_env():
        env = ObservableDeformedGridworld(
            grid_size=(1.0, 1.0),
            obstacles=obstacles,
            render_mode=render_mode,
        )

        env = Monitor(env)  # record stats such as returns
        return env

    env = DummyVecEnv([make_env])


    # update config
    run.config.update({"shear range":env.envs[0].unwrapped.shear_range,
                      "stretch range":env.envs[0].unwrapped.stretch_range}, allow_val_change=True)


    net_arch=[128, 128, 128]
    model = DQN("MultiInputPolicy",env,batch_size=batch_size,gamma=gamma, 
                target_update_interval=target_update, policy_kwargs=dict(net_arch=net_arch), verbose=1,
                tensorboard_log=f"runs/{run.id}", device="cpu", learning_rate=lr,
                train_freq=(10,"step"), gradient_steps=1)
    model.learn(total_timesteps,progress_bar=True, callback=callbacks, log_interval=55)
    model.save(f"agents/pretrained/MDP/DQN_continous_{run.id}")
    env.close()
    run.finish()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--total_timesteps", type=int, default=5000000) # env steps
    parser.add_argument("--target_update", type=int, default=200) # in env steps
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--render_mode", type=str, default=None)

    args = parser.parse_args()

    train_dqn(args)