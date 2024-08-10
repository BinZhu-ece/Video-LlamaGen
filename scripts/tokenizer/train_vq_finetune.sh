# !/bin/bash
set -x

torchrun \
--nnodes=$nnodes --nproc_per_node=$nproc_per_node --node_rank=$node_rank \
--master_addr=$master_addr --master_port=$master_port \
tokenizer/tokenizer_image/vq_train.py \
--finetune \
--disc-start 0 \
--vq-ckpt ./pretrained_models/vq_ds16_c2i.pt \
--dataset t2i_image \
--data-path /path/to/high_aesthetic_10M \
--data-face-path /path/to/face_2M \
--cloud-save-path /path/to/cloud_disk \
"$@"