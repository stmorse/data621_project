#!/bin/tcsh
#SBATCH --job-name=nri
#SBATCH -N 1 -n 4
#SBATCH --gpus=1
#SBATCH -t 12:00:00

cd ~/classes/nndl
module load python/3.12.7
source venv/bin/activate.csh
cd project

python -u train.py \
  --subfolder charged --suffix _charged5 \
  --save-folder logs/gae \
  --encoder cnn --epochs 50 \
  >& logs/gae/gae_charged_25-05-01.log
