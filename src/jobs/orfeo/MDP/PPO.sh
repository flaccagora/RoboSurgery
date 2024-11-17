#!/bin/bash
#SBATCH -A dssc
#SBATCH -p EPYC
#SBATCH --time 02:00:00     # format: HH:MM:SS
#SBATCH -N 1                # 1 node
#SBATCH --ntasks-per-node=4 # 4 tasks out of 112
#SBATCH --job-name=DQNsb3_MDP
#SBATCH --output=DQNsb3_MDP.out 

date

conda deactivate

cd ~/RoboSurgery
source .rob/bin/activate

cd ./src

export PYTHONPATH=$(pwd)

python3 train/MDP/PPO.py --learning_rate=0.001

date
