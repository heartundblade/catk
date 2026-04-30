#!/bin/sh
export LOGLEVEL=INFO
export HYDRA_FULL_ERROR=1
export TF_CPP_MIN_LOG_LEVEL=2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

MY_EXPERIMENT="vbd"
MY_TASK_NAME=$MY_EXPERIMENT"-debug"

# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate catk

torchrun \
  -m src.vbd.run_vbd \
  experiment=$MY_EXPERIMENT \
  task_name=$MY_TASK_NAME \
  trainer.limit_val_batches=2 \
  trainer.limit_train_batches=2 \
  ckpt_path=null \
  data.train_batch_size=2

# ! below is for training with ddp
# torchrun \
#   --rdzv_id $SLURM_JOB_ID \
#   --rdzv_backend c10d \
#   --rdzv_endpoint $MASTER_ADDR:$MASTER_PORT \
#   --nnodes $NUM_NODES \
#   --nproc_per_node gpu \
#   -m src.run \
#   experiment=$MY_EXPERIMENT \
#   trainer=ddp \
#   task_name=$MY_TASK_NAME

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
#   # trainer.limit_val_batches=5 \
#   # trainer.limit_train_batches=5

echo "bash train_vbd.sh done!"
