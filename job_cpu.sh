#!/bin/bash
#SBATCH -t 3-00:00
#SBATCH --account=rrg-lelliott
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=16G
#SBATCH --mail-user=<joosungm@sfu.ca>
#SBATCH --mail-type=ALL

module load python/3.7
source ~/torch/bin/activate
python "ORF_CartPole.py"

