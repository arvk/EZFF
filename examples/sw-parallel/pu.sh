#!/bin/bash
#SBATCH --partition=priya
#SBATCH --account=lc_pv
#SBATCH --ntasks=64
#SBATCH --ntasks-per-node=16
#SBATCH --constraint=IB
#SBATCH --mem-per-cpu=2GB
#SBATCH --time=02:00:00
#SBATCH --output=STREAM_OUTPUT
#SBATCH --job-name=BIGEZFF
#SBATCH --exclude=hpc3913,hpc3914,hpc3915

source /usr/usc/openmpi/default/setup.sh
source /usr/usc/python/3.6.0/setup.sh

ulimit -s unlimited

echo "starting simulation **************************************"
date
mpirun -np 64 python3 run.py
date
echo "simulation finished **************************************"
echo

exit 0
