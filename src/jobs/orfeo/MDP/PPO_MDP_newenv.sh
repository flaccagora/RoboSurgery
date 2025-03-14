#!/bin/bash
#SBATCH -A dssc
#SBATCH -p EPYC
#SBATCH --time 02:00:00     # format: HH:MM:SS
#SBATCH -N 1                # 1 node
#SBATCH --ntasks-per-node=24 # 4 tasks out of 112
#SBATCH --job-name=PPO_MDP
#SBATCH --output=PPO_MDP.out 

date

conda deactivate

cd ~/RoboSurgery
source .rob/bin/activate

cd ./src

export PYTHONPATH=$(pwd)

python3 train/MDP/PPO_newenv.py --total_timesteps=20000000 --learning_rate=0.0005

date
