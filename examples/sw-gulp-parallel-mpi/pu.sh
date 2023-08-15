#!/bin/bash
#SBATCH --ntasks=32
#SBATCH --time=04:00:00
#SBATCH --output=STREAM_OUTPUT
#SBATCH --job-name=SW-PARALLEL

srun -n 32 --mpi=pmi2 python3.7 run.py

exit 0
