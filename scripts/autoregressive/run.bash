# !/bin/bash
export CUDA_VISIBLE_DEVICES=0
export master_addr=127.0.0.1
export master_port=29501

set -x

torchrun \
--nnodes=1 --nproc_per_node=1  \
--master_addr=$master_addr --master_port=$master_port \
autoregressive/train/train_t2i.py \
--vq-ckpt ./pretrained_models/vq_ds16_t2i.pt \
--data-path /path/to/laion_coco50M \
--t5-feat-path /path/to/laion_coco50M_flan_t5_xl \
--dataset t2i \
--image-size 256 \
"$@"
