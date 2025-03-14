from stable_baselines3 import DQN
import numpy as np
from environment.env import MDPGYMGridEnvDeform
import numpy as np


# thetas deformations (range(a,b),range(c,d))
l0 = 1
h0 = 10
l1 = 1
h1 = 10

def train_dqn(args):
    from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
    from wandb.integration.sb3 import WandbCallback
    import wandb

    total_timesteps = args.total_timesteps
    batch_size = args.batch_size
    lr = args.learning_rate
    target_update = args.target_update
    gamma = args.gamma

    config = {
        "policy_type": "MultiInputPolicy",
        "env_name": "MDPGYMGridEnvDeform",
        "defo_range": (l0,h0,l1,h1),
        "total_timesteps": total_timesteps,
        "learning_rate": lr,
        "gamma": args.gamma,
        "target_update": target_update,
        "Batch_Size": batch_size,
        }

    run = wandb.init(
        project="DQNsb3 - MDP",
        config=config,
        sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
        monitor_gym=True,  # auto-upload the videos of agents playing the game
        save_code=True,  # optional
    )


    # Save a checkpoint every 10000 steps
    checkpoint_callback = CheckpointCallback(
                            save_freq=10000,
                            save_path=f"agents/pretrained/MDP/DQNsb3_{run.id}",
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
    from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder, SubprocVecEnv

    def make_env():
        N = 2

        # thetas deformations (range(a,b),range(c,d))
        l0 = 1
        h0 = 10
        l1 = 1
        h1 = 10
        
        maze = np.load(f"maze/maze_{N}.npy")
        env = MDPGYMGridEnvDeform(maze,l0,h0,l1,h1, render_mode="sbaruba")

        env = Monitor(env)  # record stats such as returns
        return env

    # env = DummyVecEnv([make_env])
    env = SubprocVecEnv([make_env]  * 20)


    model = DQN("MultiInputPolicy",env,batch_size=batch_size,gamma=gamma, 
                target_update_interval=target_update,verbose=2,
                tensorboard_log=f"runs/{run.id}", device="cpu", learning_rate=lr,
                train_freq=(1,"step"), gradient_steps=1)
    model.learn(total_timesteps,progress_bar=True, callback=callbacks, log_interval=1)
    model.save(f"agents/pretrained/MDP/DQNsb3_{run.id}")
    env.close()
    run.finish()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--total_timesteps", type=int, default=1000000) # env steps
    parser.add_argument("--target_update", type=int, default=200) # in env steps
    parser.add_argument("--gamma", type=float, default=0.99)

    args = parser.parse_args()

    train_dqn(args)