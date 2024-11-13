#!/bin/bash

#SBATCH --job-name=NN
#SBATCH --nodes=1
#SBATCH --time=2:00:00
#SBATCH --partition=EPYC
#SBATCH --output=QL-%j.out

date

cd ~/RoboSurgery
source ~/miniconda3/bin/activate
conda activate rob

python3 train.py

date
