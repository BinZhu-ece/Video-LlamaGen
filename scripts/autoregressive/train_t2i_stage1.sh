# !/bin/bash
set -x


export master_addr=127.0.0.1
export master_port=29504
# export CUDA_VISIBLE_DEVICES=0
# export CUDA_LAUNCH_BLOCKING=1
cd /storage/zhubin/LlamaGen 
source  /storage/miniconda3/etc/profile.d/conda.sh 
conda activate motionctrl


# export TORCH_LOGS="+dynamo"
# export TORCHDYNAMO_VERBOSE=1


CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4  \
--master_addr=$master_addr --master_port=$master_port \
autoregressive/train/train_t2i.py \
--vq-ckpt ./pretrained_models/vq_ds16_t2i.pt \
--data-path /storage/zhubin/LlamaGen/dataset/Image_Datasets/  \
--t5-feat-path /storage/zhubin/LlamaGen/dataset/Image_Datasets/  \
--dataset t2i \
--image-size 256 \
--cloud-save-path ./cloud_path  \
--global-batch-size 24 \
--log-every 10 \
--epochs 300 \
--ckpt-every 1000 
"$@"
