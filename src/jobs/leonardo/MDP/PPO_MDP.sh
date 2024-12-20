#!/bin/bash
#SBATCH -A ICT24_DSSC_CPU
#SBATCH -p dcgp_usr_prod
#SBATCH --time 04:00:00     # format: HH:MM:SS
#SBATCH -N 1                # 1 node
#SBATCH --ntasks-per-node=8 # 4 tasks out of 112
#SBATCH --mem=128000          # memory per node out of 494000MB (481GB)
#SBATCH --job-name=PPO_MDP
#SBATCH --output=PPO_MDP.out 

date

cd ~/RoboSurgery
source .rob/bin/activate

cd ./src

export PYTHONPATH=$(pwd)
export WANDB_MODE=offline

python3 train/MDP/PPO_newenv.py --total_timesteps=20000000 
date
