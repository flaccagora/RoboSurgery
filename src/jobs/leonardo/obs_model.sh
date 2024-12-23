#!/bin/bash
#SBATCH -A ICT24_DSSC_GPU
#SBATCH -p boost_usr_prod
#SBATCH --time 04:00:00     # format: HH:MM:SS
#SBATCH -N 1                # 1 node
#SBATCH --ntasks-per-node=8 # 4 tasks out of 112
#SBATCH --mem=64000          # memory per node out of 494000MB (481GB)
#SBATCH --gres=gpu:1
#SBATCH --job-name=obs
#SBATCH --output=obs.out 

date

cd ~/RoboSurgery
source .rob/bin/activate

cd ./src

export PYTHONPATH=$(pwd)
export WANDB_MODE=offline

python3 train/train_obs_model.py 
date
