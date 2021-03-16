#!/bin/bash

# to run single experiment
# Slurm sbatch options
#SBATCH -a 0-7
#SBATCH --gres=gpu:volta:1
#SBATCH -n 10

# Loading the required module
source /etc/profile
module load anaconda/2020a 

mkdir -p wandb_gnn
# Run the script
polars=('True' 'True' 'True' 'True' 'False' 'False' 'False' 'False')
nodes=(50 100 200 300 50 100 200 300)
# script to iterate through different hyperparameters
python main.py --num_nodes=${nodes[$SLURM_ARRAY_TASK_ID]} --polar=${polars[$SLURM_ARRAY_TASK_ID]} --batch_size=64 &> out_${nodes[$SLURM_ARRAY_TASK_ID]}_${polars[$SLURM_ARRAY_TASK_ID]}