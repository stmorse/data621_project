#!/bin/tcsh
#SBATCH --job-name=simgen 
#SBATCH -N 1 -n 4
#SBATCH -t 24:00:00

cd ~/classes/nndl
module load python/3.12.7
source venv/bin/activate.csh

cd project/data
python -u generate_dataset.py \
  --subfolder charged --simulation charged \
  --num-train 50000 --num-valid 10000 --num-test 10000 \
  >& output_charged.log
