#!/bin/bash
#SBATCH -p long
#SBATCH --time 02:00:00     # format: HH:MM:SS
#SBATCH -N 1                # 1 node
#SBATCH --ntasks-per-node=4 # 4 tasks out of 112
#SBATCH --job-name=DQN_MDP
#SBATCH --output=DQN_MDP.out 

date

conda deactivate

cd ~/RoboSurgery
source .rob/bin/activate

cd ./src

export PYTHONPATH=$(pwd)

python3 train/MDP/DQN_newenv.py --learning_rate=0.001

date
