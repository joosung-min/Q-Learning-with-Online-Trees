#!/bin/bash
#SBATCH -t 14-00:00
#SBATCH --account=rrg-lelliott
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --mail-user=<joosungm@sfu.ca>
#SBATCH --mail-type=ALL

module load python/3.8
source ~/torch/bin/activate
python setup_ORF.py build_ext --inplace
python "ORF_mtcar.py"

