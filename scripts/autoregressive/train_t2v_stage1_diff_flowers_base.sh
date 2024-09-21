# !/bin/bash

cd  /storage/zhubin/LlamaGen
DATA_FILE='/storage/zhubin/LlamaGen/dataset/Image_Datasets/flowers/meta_data.json'
# CKPT=/storage/zhubin/LlamaGen/CausalVideoVAE/vae_ckpt/432322048
CKPT=/storage/zhubin/LlamaGen/CausalVideoVAE/vae_ckpt/488dim8

nnodes=1
nproc_per_node=8
export master_addr=127.0.0.1
export master_port=29506
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

source  /storage/miniconda3/etc/profile.d/conda.sh 
conda activate motionctrl


torchrun \
--nnodes=$nnodes --nproc_per_node=$nproc_per_node  \
--master_addr=$master_addr --master_port=$master_port \
autoregressive/train/train_t2v.py \
--vae-model  VAE-16 \
--vae-ckpt ${CKPT} \
--data-path None \
--enable_tiling \
--tile_overlap_factor 0.25 \
--image-size 256 \
--video_meta_info_file $DATA_FILE \
--t5-model-path  pretrained_models/t5-ckpt \
--t5-model-type  flan-t5-xl \
--model_max_length 512 \
--num_frames  1  \
--cloud-save-path ./cloud_path_t2v  \
--global-batch-size $(( 32 * $nproc_per_node )) \
--max_height 256 \
--max_width 256 \
--epochs  1000 \
--gpt-type t2v \
--dataset t2v \
--num-workers 24  \
--log-every 1  \
--ckpt-every  10000  \
--results-dir results_vae_1f_diff_flowers_GPT-B \
--start_frame_ind 1 \
--data_root /storage/zhubin/LlamaGen/dataset/Image_Datasets/flowers \
--t5-path  /storage/zhubin/LlamaGen/dataset/storage_datasets_npy/flowers \
--gpt-model GPT-B \
--gradient-accumulation-steps 1 \
--prefetch_factor 4 \
--downsample-size 8 \
--mixed-precision none \
--no-compile  \
--data_repeat 10 






# bs=2  frames=1 17G
# bs=8  frames=1  50G
# bs=12 frames=1   59G  GPU6
# bs=16 frames=1   78G   GPU5
 
 