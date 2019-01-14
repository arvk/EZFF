#!/bin/bash
#SBATCH --ntasks=16
#SBATCH --time=04:00:00
#SBATCH --output=STREAM_OUTPUT
#SBATCH --job-name=LJ-PARALLEL

srun -n 16 --mpi=pmi2 python3.7 run.py

exit 0
