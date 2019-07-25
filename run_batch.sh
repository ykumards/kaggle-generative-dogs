#!/bin/bash
#SBATCH --time=3:00:00
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=32G

module load anaconda3
pip install pytorch-ignite --user
srun ipython notebooks/dcgan-ignite-baseline-leval.py 
