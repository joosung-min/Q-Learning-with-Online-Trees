#!/bin/bash
#SBATCH -t 3-00:00
#SBATCH --account=rrg-lelliott
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=16G
#SBATCH --mail-user=<joosungm@sfu.ca>
#SBATCH --mail-type=ALL

module load python/3.8
source ~/torch/bin/activate
python "ORF_LunarLander_v1.py"

