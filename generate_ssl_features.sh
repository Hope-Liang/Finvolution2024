#!/bin/bash
#SBATCH --gpus=1
#SBATCH --time=24:00:00
#SBATCH --mail-type=END
#SBATCH --mail-user=fcumlin@gmail.com

python generate_ssl_features.py --model="w2v2_xlsr_2b" --folder="finvcup9th_1st_ds5" --split="train" --target_duration=8 --batch=${SLURM_ARRAY_TASK_ID} --num_batches=$1 # FinvCup data
python generate_ssl_features_es.py --model="w2v2_xlsr_2b" --folder="NOT_USED" --split="train" --target_duration=8 --batch=${SLURM_ARRAY_TASK_ID} --num_batches=$1 # ES data
#python generate_ssl_features.py --model="w2v2_xlsr_2b" --folder="finvcup9th_2nd_ds2a" --split="test" --target_duration=8  --batch=${SLURM_ARRAY_TASK_ID} --num_batches=$1 # FinvCup data
#python generate_ssl_features_cfad.py --model="mms-1b" --target_duration=4 --batch=${SLURM_ARRAY_TASK_ID} --num_batches=$1  # CFAD real
#python generate_ssl_features_asv.py --model="mms-1b" --target_duration=4 --batch=${SLURM_ARRAY_TASK_ID} --num_batches=$1  # ASVSpoof