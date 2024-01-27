#!/usr/bin/env bash
# File       : run.sh
# Description: Parallelized adversarial model training.
#              ./run.sh

#SBATCH --job-name=adv_training_parallel
#SBATCH --output=adv_training_parallel_%j.out
#SBATCH --error=adv_training_parallel_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --partition=academic
#SBATCH --time=04:00:00

module purge
module load gcc/10.2.0-fasrc01 openmpi/4.1.1-fasrc01 CMake/3.9.1

cd build

# 2. run training
for advsteps in 1 2 4 8 16 32 64 ; do
    srun -n$advsteps ./mnist 10 $advsteps 1.0 0
done
