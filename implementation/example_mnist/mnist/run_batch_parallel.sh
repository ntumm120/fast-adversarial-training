#!/usr/bin/env bash
for advsteps in 2 4 8 ; do
sbatch <<EOT
#!/usr/bin/env bash
# File       : run_batch_parallel.sh
# Description: Parallelized adversarial model training.
#SBATCH --job-name=adv_training_parallel_"$advsteps"
#SBATCH --output=adv_training_parallel_"$advsteps"_%j.out
#SBATCH --error=adv_training_parallel_"$advsteps"_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node="$advsteps"
#SBATCH --cpus-per-task=1
#SBATCH --partition=broadwell
#SBATCH --time=01:00:00
module purge
module load gcc/10.2.0-fasrc01 openmpi/4.1.1-fasrc01 CMake/3.9.1
cd build
# 2. run training
srun ./mnist 3 "$advsteps" 1.0 1 1
EOT
done
