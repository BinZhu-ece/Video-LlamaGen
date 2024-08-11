# !/bin/bash
set -x

torchrun \
--nnodes=$nnodes --nproc_per_node=$nproc_per_node --node_rank=$node_rank \
--master_addr=$master_addr --master_port=$master_port \
autoregressive/train/train_t2i.py \
--vq-ckpt ./pretrained_models/vq_ds16_t2i.pt \
--data-path /storage/zhubin/LlamaGen/dataset/Image_Datasets/Image_civitai_10000 \
--t5-feat-path /storage/zhubin/LlamaGen/dataset/Image_Datasets/civitai_10000_flan_t5_xl \
--dataset t2i \
--image-size 256 \
"$@"
