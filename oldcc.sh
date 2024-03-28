#!/bin/bash
### GPU OPTIONS:
### CEDAR: v100l, p100
### BELUGA: *no option, just use --gres=gpu:*COUNT*
### GRAHAM: v100, t4
### see https://docs.computecanada.ca/wiki/Using_GPUs_with_Slurm
#SBATCH --gres=gpu:v100l:1
#SBATCH --nodes=1
#SBATCH --exclude=cdr897
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --mem-per-cpu=8000
#SBATCH --time=00:10:00
#SBATCH --mail-user=1835928575qq@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --output=output/output-%j.txt

CUDA_VISIBLE_DEVICES=0 python3 train.py