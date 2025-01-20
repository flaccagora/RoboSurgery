#!/bin/bash
#SBATCH -A dssc
#SBATCH -p GPU
#SBATCH --time 01:00:00     # format: HH:MM:SS
#SBATCH -N 1                # 1 node
#SBATCH --ntasks-per-node=8 # 4 tasks out of 112
#SBATCH --mem=64000          # memory per node out of 494000MB (481GB)
#SBATCH --job-name=obs
#SBATCH --output=obs.out 
#SBATCH --gres=gpu:1

date

cd ~/RoboSurgery
source .rob/bin/activate

cd ./src

export PYTHONPATH=$(pwd)

python3 train/train_vae.py 
date
