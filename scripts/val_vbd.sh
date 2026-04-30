#!/bin/sh
export LOGLEVEL=INFO
export HYDRA_FULL_ERROR=1
export TF_CPP_MIN_LOG_LEVEL=2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

MY_EXPERIMENT="vbd"
MY_TASK_NAME=$MY_EXPERIMENT"-val-debug"

# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate catk

python \
  -m src.vbd.run_vbd \
  experiment=$MY_EXPERIMENT \
  trainer=default \
  trainer.accelerator=gpu \
  trainer.devices=1 \
  trainer.strategy=auto \
  task_name=$MY_TASK_NAME \
  action=validate \
  trainer.limit_val_batches=2 \
  data.val_batch_size=3

# torchrun \
#   -m src.vbd.run_vbd \
#   experiment=$MY_EXPERIMENT \
#   task_name=$MY_TASK_NAME \
#   trainer.limit_val_batches=2 \
#   trainer.limit_train_batches=2

# torchrun \
#   --rdzv_id 12345 \
#   --rdzv_backend c10d \
#   --rdzv_endpoint localhost:29500 \
#   --nnodes 1 \
#   --nproc_per_node 7 \
#   -m src.vbd.run_vbd \
#   experiment=$MY_EXPERIMENT \
#   trainer=ddp \
#   task_name=$MY_TASK_NAME # \
  # trainer.limit_val_batches=5 \
  # trainer.limit_train_batches=5

echo "bash val_vbd.sh done!"
