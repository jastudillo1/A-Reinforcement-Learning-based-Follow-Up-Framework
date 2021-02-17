#!/bin/bash
#SBATCH -n 24                # Number of cores
#SBATCH -N 1                  # Number of cores
#SBATCH -t 0-4:00              # Runtime in D-HH:MM, minimum of 10 minutes
#SBATCH -p shared               # Partition to submit to
#SBATCH --mem=12000       # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH -o %j.out               # File to which STDOUT will be written, %j inserts jobid
#SBATCH -e %j.err               # File to which STDERR will be written, %j inserts jobid

module load Anaconda3/5.0.1-fasrc02
source activate default
python -u ./job.py
