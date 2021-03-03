#!/bin/bash
#SBATCH -t 6:00:00
#SBATCH --account=rrg-lelliott
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=1G
#SBATCH --mail-user=<joosungm@sfu.ca>
#SBATCH --mail-type=ALL
#SBATCH --array 1-5

module load python/3.8
source ~/torch/bin/activate
python "test.py"
