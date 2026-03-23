#!/bin/sh
export LOGLEVEL=INFO
export HYDRA_FULL_ERROR=1
export TF_CPP_MIN_LOG_LEVEL=2

MY_EXPERIMENT="local_val"
VAL_K=48
MY_TASK_NAME=$MY_EXPERIMENT-K$VAL_K"-debug"

# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate catk

# local_val runs on single GPU
python \
  -m src.run \
  experiment=$MY_EXPERIMENT \
  trainer=default \
  model.model_config.validation_rollout_sampling.num_k=$VAL_K \
  trainer.accelerator=gpu \
  trainer.devices=1 \
  trainer.strategy=auto \
  task_name=$MY_TASK_NAME # \
  # trainer.limit_val_batches=5 \
  # trainer.limit_train_batches=5

torchrun \
  --rdzv_id 12345 \
  --rdzv_backend c10d \
  --rdzv_endpoint localhost:29500 \
  --nnodes 1 \
  --nproc_per_node 4 \
  -m src.run \
  experiment=$MY_EXPERIMENT \
  trainer=ddp \
  task_name=$MY_TASK_NAME \
  model.model_config.validation_rollout_sampling.num_k=$VAL_K
  # trainer.limit_val_batches=5 \
  # trainer.limit_train_batches=5

echo "bash local_val.sh done!"