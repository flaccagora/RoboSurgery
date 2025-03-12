import numpy as np
import gymnasium as gym
from environment.env import Grid
import os
import re


from stable_baselines3 import DQN
import numpy as np

obstacles = [((0.14625, 0.3325), (0.565, 0.55625)), 
             ((0.52875, 0.5375), (0.7375, 0.84125)), 
             ((0.0, 0.00125), (0.01625, 0.99125)), 
             ((0.0075, 0.00125), (0.99875, 0.04)), 
             ((0.98875, 0.0075), (0.99875, 1.0)), 
             ((0.00125, 0.9825), (0.99875, 1.0))]

def find_last_checkpoint(directory):
    # Define the regex pattern to match the checkpoint file names
    pattern = re.compile(r"rl_model_(\d+)_steps")

    last_checkpoint = None
    max_steps = -1

    # Iterate through all files in the directory
    for filename in os.listdir(directory):
        match = pattern.match(filename)
        if match:
            # Extract the step number
            steps = int(match.group(1))
            # Check if this is the highest step count seen so far
            if steps > max_steps:
                max_steps = steps
                last_checkpoint = filename

    return os.path.splitext(last_checkpoint)[0] if last_checkpoint else None



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
    run_id = args.run_id

    n_envs = 20

    config = {
        "policy_type": "MultiInputPolicy",
        "env_name": "ObservableDeformedGridworld",
        "total_timesteps": total_timesteps,
        "Batch_Size": batch_size,
        'grid_size': (1.0,1.0),
        'step_size': 0.02,
        'obstacles':obstacles,
        'observation_radius':0.2,
        "shear range":None,
        "stretch range":None
    }
    if run_id is not None:
        run = wandb.init(
            project="DQNsb3 - MDP - ObservableDeformedGridworld",
            config=config,
            sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
            monitor_gym=True,  # auto-upload the videos of agents playing the game
            save_code=True,  # optional
            id=run_id,
            resume="must",
        )
    else:
        run = wandb.init(
            project="DQNsb3 - MDP - ObservableDeformedGridworld",
            config=config,
            sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
            monitor_gym=True,  # auto-upload the videos of agents playing the game
            save_code=True,  # optional
        )


    # Save a checkpoint every 10000 steps
    checkpoint_callback = CheckpointCallback(
                            save_freq= 250000 // n_envs, # in steps
                            save_path=f"agents/pretrained/MDP/DQN_continous_{run.id}",
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
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

    def make_env():
        env = Grid(
            grid_size=(1.0, 1.0),
            obstacles=obstacles,
            render_mode=render_mode,
            shear_range=(-.2,.2),
            stretch_range=(.4,1)
        )

        env = Monitor(env)  # record stats such as returns
        return env

    # env = DummyVecEnv([make_env])
    env = SubprocVecEnv([make_env] * n_envs)

    # # update config
    # run.config.update({"shear range":env.envs[0].unwrapped.shear_range,
    #                   "stretch range":env.envs[0].unwrapped.stretch_range}, allow_val_change=True)


    net_arch=[128, 128, 128]
    if args.run_id is not None:
        last_checkpoint = find_last_checkpoint(f"agents/pretrained/MDP/DQN_continous_{args.run_id}")
        model = DQN.load(f"./agents/pretrained/MDP/DQN_continous_{args.run_id}/{last_checkpoint}",env=env,)
        print(f"agents/pretrained/MDP/DQN_continous_{args.run_id}/{last_checkpoint}")
    else:
        model = DQN("MultiInputPolicy",env,batch_size=batch_size,gamma=gamma, 
                target_update_interval=target_update, policy_kwargs=dict(net_arch=net_arch), verbose=1,
                tensorboard_log=f"runs/{run.id}", device="cpu", learning_rate=lr,
                train_freq=(5,"step"), gradient_steps=1, exploration_fraction=0.01)
    
    model.learn(total_timesteps,progress_bar=True, callback=callbacks, log_interval=55,reset_num_timesteps=False)
    model.save(f"agents/pretrained/MDP/DQN_continous_{run.id}")
    env.close()
    run.finish()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--total_timesteps", type=int, default=5000000) # env steps
    parser.add_argument("--target_update", type=int, default=1000) # in env steps
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--render_mode", type=str, default=None)
    parser.add_argument("--run_id", type=str, default=None)

    args = parser.parse_args()

    train_dqn(args)