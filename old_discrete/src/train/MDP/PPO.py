from stable_baselines3 import PPO
import numpy as np
from environment.env import MDPGYMGridEnvDeform
import numpy as np


# thetas deformations (range(a,b),range(c,d))
l0 = 1
h0 = 10
l1 = 1
h1 = 10

def train_ppo(args):
    from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
    from wandb.integration.sb3 import WandbCallback
    import wandb

    class My_callback(BaseCallback):
        def __init__(self, verbose=0):
            super(My_callback, self).__init__(verbose)
        def _on_step(self) -> bool:
            if self.num_timesteps % 200 == 0:
                self.training_env.reset()
            return True
        def _on_rollout_end(self) -> None:
            print(f"Rollout end: {self.num_timesteps}")
            return True
        

    total_timesteps = args.total_timesteps
    batch_size = args.batch_size
    n_steps = args.n_steps
    lr = args.learning_rate

    config = {
        "policy_type": "MultiInputPolicy",
        "env_name": "MDPGYMGridEnvDeform",
        "defo_range": (l0,h0,l1,h1),
        "total_timesteps": total_timesteps,
        "Batch_Size": batch_size,
        "PPO n_steps": n_steps
    }

    run = wandb.init(
        project="PPO - MDP",
        config=config,
        sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
        monitor_gym=True,  # auto-upload the videos of agents playing the game
        save_code=True,  # optional
    )

    # Save a checkpoint every 1000 steps
    checkpoint_callback = CheckpointCallback(
                            save_freq=500000,
                            save_path=f"agents/pretrained/POMDP/PPO_{run.id}",
                            name_prefix="rl_model",
                            save_replay_buffer=False,
                            save_vecnormalize=True,
                        )


    callbacks = [
                WandbCallback(gradient_save_freq=100,
                                model_save_path=f"agents/pretrained/MDP/PPO_{run.id}",
                                verbose=2,
                                model_save_freq = total_timesteps//10
                                ),
                checkpoint_callback,
                ]

    # n_steps (int) – The number of steps to run for each environment per update 
    # (i.e. rollout buffer size is n_steps * n_envs
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder

    def make_env():
        N = 2

        # thetas deformations (range(a,b),range(c,d))
        l0 = 1
        h0 = 10
        l1 = 1
        h1 = 10
        
        maze = np.load(f"maze/maze_{N}.npy")
        env = MDPGYMGridEnvDeform(maze,l0,h0,l1,h1, render_mode="rgb_array")

        env = Monitor(env)  # record stats such as returns
        return env

    env = DummyVecEnv([make_env])


    model = PPO("MultiInputPolicy",env,n_steps=n_steps,batch_size=batch_size,verbose=1,tensorboard_log=f"runs/{run.id}", device="cpu", learning_rate=lr)
    model.learn(total_timesteps,progress_bar=True, callback=callbacks)
    model.save(f"agents/pretrained/MDP/PPO_{run.id}")
    env.close()
    run.finish()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--learning_rate", type=float, default=0.0003)
    parser.add_argument("--batch_size", type=int, default=2000)
    parser.add_argument("--n_steps", type=int, default=2000)
    parser.add_argument("--total_timesteps", type=int, default=10000000)
    
    args = parser.parse_args()

    train_ppo(args)