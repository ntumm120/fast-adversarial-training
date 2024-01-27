cd build

make

# Strong Scaling, advsteps = 32
echo "Running strong scaling!"
advsteps=32
for processes in 1 2 4 8 16 32 ; do
    mpirun -np $processes ./mnist 1 $advsteps 1.0 0 1 >> ../results/strong_scaling.txt
done

# Weak Scaling, advsteps = num_processes
echo "Running weak scaling"
for processes in 1 2 4 8 16 32 ; do
    advsteps="$(($processes-1))"
    # We set number of adversary steps at test time to 0 to run experiments fast
    mpirun -np $processes ./mnist 3 $advsteps 1.0 0 1 10 0 >> ../results/weak_scaling.txt
done

# PyTorch Intra-op (32 cores)
# for advsteps in 1 2 4 8 16 32 64 ; do
#     mpirun ./mnist 10 $advsteps 1.0 0 0 >> seq_bound.txt
# done

# Staleness
# advsteps=32
# for staleness in 1 2 4 8 16 32 ; do
#     mpirun -np $advsteps ./mnist 10 $advsteps 1.0 0 $staleness >> ../results/staleness.txt
# done



