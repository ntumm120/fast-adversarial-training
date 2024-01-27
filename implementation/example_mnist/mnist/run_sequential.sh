#!/usr/bin/env bash
# File       : run_sequential.sh
# Description: Sequential adversarial model training.

#SBATCH --job-name=adv_training_sequential
#SBATCH --output=adv_training_sequential_%j.out
#SBATCH --error=adv_training_sequential_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=broadwell
#SBATCH --time=02:00:00
module purge
module load gcc/10.2.0-fasrc01 openmpi/4.1.1-fasrc01 CMake/3.9.1
cd build
# 2. run training
for advsteps in 1 2 4 8 16 32 ; do
    srun ./mnist 1 $advsteps 1.0 0 1
done
