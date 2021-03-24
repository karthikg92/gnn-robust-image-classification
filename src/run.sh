#!/bin/bash

# to run single experiment
# Slurm sbatch options
#SBATCH -a 0-14
#SBATCH --gres=gpu:volta:1
#SBATCH -n 10

# Loading the required module
source /etc/profile
module load anaconda/2020a 

mkdir -p wandb_gnn
mkdir -p out_files
# Run the script
nodes=(200 300 400 500 600 200 300 400 500 600 200 300 400 500 600)
Ks=(2 2 2 2 2 3 3 3 3 3 5 5 5 5 5)
# script to iterate through different hyperparameters
python main.py --num_nodes=${nodes[$SLURM_ARRAY_TASK_ID]} --K=${Ks[$SLURM_ARRAY_TASK_ID]} --dataset='CIFAR' --batch_size=64 &> out_files/out_${nodes[$SLURM_ARRAY_TASK_ID]}_${Ks[$SLURM_ARRAY_TASK_ID]}_CIFAR