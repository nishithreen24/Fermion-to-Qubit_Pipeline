#!/bin/bash
#SBATCH -J xbk_siam
#SBATCH -o Out_%j.txt
#SBATCH -e Err_%j.txt
#SBATCH -p kshort
#SBATCH --mail-user=<username>@imsc.res.in
#SBATCH --mail-type=ALL
#SBATCH --mem-per-cpu=4000
#SBATCH -n 16
#SBATCH -c 1
#SBATCH -t 0-12:00:00
#SBATCH --export=all
#SBATCH -D <working_directory>

echo $SLURM_JOB_NODELIST > hostfile_$SLURM_JOBID

export OMP_NUM_THREADS=1

module purge
module try-load python/3.10
module try-load openmpi/4.1

mpirun -np 16 python runthis.py
