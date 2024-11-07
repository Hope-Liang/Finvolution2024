#!/bin/bash
#SBATCH --gpus=1
#SBATCH --time=24:00:00

python train.py --gin_path "configs/tot.gin" --features_folder="train_feature_w2v2_xlsr_2b_layer${SLURM_ARRAY_TASK_ID}" --save_path "runs/w2v_8s_es_layer${SLURM_ARRAY_TASK_ID}"