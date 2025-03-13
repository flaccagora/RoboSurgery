#!/bin/bash
#SBATCH -A dssc
#SBATCH -p THIN
#SBATCH --time 02:00:00     # format: HH:MM:SS
#SBATCH -N 1                # 1 node
#SBATCH --ntasks=12 # 4 tasks out of 112
#SBATCH --job-name=eval
#SBATCH --output=eval.out 

date

cd ~/RoboSurgery
source .rob/bin/activate

cd ./src

export PYTHONPATH=$(pwd)

python3 eval/eval.py

date
