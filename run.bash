# !/bin/bash


#================ 第一步先提取  图片的caption T5特征 ================

export CUDA_VISIBLE_DEVICES=7
/storage/zhubin/LlamaGen/scripts/language/extract_flan_t5_feat_laion_coco_stage1.sh







set -x

export CUDA_VISIBLE_DEVICES=7
export master_addr=127.0.0.1
export master_port=29501
export CUDA_VISIBLE_DEVICES=7
conda activate motionctrl
torchrun \
--nnodes=1 --nproc_per_node=1  \
--master_addr=$master_addr --master_port=$master_port \
autoregressive/train/train_t2i.py \
--vq-ckpt ./pretrained_models/vq_ds16_t2i.pt \
--data-path /storage/zhubin/LlamaGen/dataset/Image_Datasets/  \
--t5-feat-path /storage/zhubin/LlamaGen/dataset/Image_Datasets/  \
--dataset t2i \
--image-size 256 \
--cloud-save-path ./cloud_path  \
--global-batch-size 256 \
--log-every 10 \
--epochs 300 \
--ckpt-every 1000 
# --log-every 
# --num-workers 24


"$@"
