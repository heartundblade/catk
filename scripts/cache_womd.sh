#!/bin/sh
export LOGLEVEL=INFO
export HYDRA_FULL_ERROR=1
export TF_CPP_MIN_LOG_LEVEL=2

DATA_SPLIT=training # training, validation, testing

# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate catk
# --output_dir /scratch/cache/SMART
python \
  -m src.data_preprocess \
  --split $DATA_SPLIT \
  --num_workers 2 \
  --input_dir /inspire/dataset/waymo/motion-v-1-3-0/waymo_open_dataset_motion_v_1_3_0/scenario \
  --output_dir /inspire/hdd/project/fengsiyuan/maoyingming-253208110273/zhl/dataset/womd