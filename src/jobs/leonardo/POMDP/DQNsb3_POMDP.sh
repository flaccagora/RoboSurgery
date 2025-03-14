#!/bin/bash
#SBATCH -A ICT24_DSSC_CPU
#SBATCH -p dcgp_usr_prod
#SBATCH --time 02:00:00     # format: HH:MM:SS
#SBATCH -N 1                # 1 node
#SBATCH --ntasks-per-node=4 # 4 tasks out of 112
#SBATCH --mem=64000          # memory per node out of 494000MB (481GB)
#SBATCH --job-name=DQNsb3_POMDP
#SBATCH --output=DQNsb3_POMDP.out 

date

cd ~/RoboSurgery
source .rob/bin/activate

cd ./src

export PYTHONPATH=$(pwd)
export WANDB_MODE=offline

python3 train/POMDP/DQN_sb3.py

date
