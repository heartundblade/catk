#!/bin/sh
export LOGLEVEL=INFO
export HYDRA_FULL_ERROR=1
export TF_CPP_MIN_LOG_LEVEL=2

DATA_SPLIT=validation # training, validation, testing

# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate catk
# --output_dir /scratch/cache/SMART
python \
  -m src.vbd.data_preprocess.data_preprocess \
  --split $DATA_SPLIT \
  --num_workers 2 \
  --input_dir ~/Repos/SMART/data/waymo/scenario \
  --output_dir /home/zhanghailiang/Repos/catk/data_new