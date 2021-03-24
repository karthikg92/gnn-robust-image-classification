#!/bin/bash

# to run single experiment
# Slurm sbatch options
#SBATCH -a 0-8
#SBATCH --gres=gpu:volta:1
#SBATCH -n 10

# Loading the required module
source /etc/profile
module load anaconda/2020a 

mkdir -p wandb_gnn
# Run the script
nodes=(500 500 500 600 600 600 700 700 700)
Ks=(2 3 5 2 3 5 2 3 5)
# script to iterate through different hyperparameters
python main.py --num_nodes=${nodes[$SLURM_ARRAY_TASK_ID]} --K=${Ks[$SLURM_ARRAY_TASK_ID]} --remove_9='False' --batch_size=64 &> out_${nodes[$SLURM_ARRAY_TASK_ID]}_${Ks[$SLURM_ARRAY_TASK_ID]}