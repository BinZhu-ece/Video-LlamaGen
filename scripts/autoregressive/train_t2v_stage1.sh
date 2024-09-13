# !/bin/bash

cd  /storage/zhubin/LlamaGen

# DATA_FILE='/storage/zhubin/liuyihang/add_aes/output/sucai_aes_1000.json'
DATA_FILE='/storage/zhubin/video_statistics_data/task1.5/Final_format_dataset_data_v2/step1.5_istock_final_815070_random_3000.json'


CKPT=/storage/zhubin/LlamaGen/CausalVideoVAE/vae_ckpt/432322048
nnodes=1
nproc_per_node=8
# export CUDA_VISIBLE_DEVICES=7
export master_addr=127.0.0.1
export master_port=29505
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
--start_frame_ind 25 \
--num_frames 1 \
--data_root  /storage/dataset  \
--cloud-save-path ./cloud_path_t2v  \
--global-batch-size $(( 12 * $nproc_per_node )) \
--max_height 256 \
--max_width 256 \
--epochs  3000 \
--t5-path  /storage/zhubin/LlamaGen/dataset/storage_datasets_npy \
--gpt-type t2v \
--dataset t2v \
--num-workers 8  \
--log-every 1  \
--ckpt-every  3000  \
--results-dir results_vae_1f_shuffle



# bs=2  frames=1 17G
# bs=8  frames=1  50G
# bs=12 frames=1   59G  GPU6
# bs=16 frames=1   78G   GPU5
 

 # debug and save_train_video_latent

cd  /storage/zhubin/LlamaGen
DATA_FILE='/storage/zhubin/video_statistics_data/task1.5/Final_format_dataset_data_v2/step1.5_istock_final_815070_random_3000.json'
CKPT=/storage/zhubin/LlamaGen/CausalVideoVAE/vae_ckpt/432322048
nnodes=1
nproc_per_node=1
export master_addr=127.0.0.1
export master_port=29509
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
--start_frame_ind 25 \
--num_frames 1 \
--data_root  /storage/dataset  \
--cloud-save-path ./cloud_path_t2v  \
--global-batch-size $(( 12 * $nproc_per_node )) \
--max_height 256 \
--max_width 256 \
--epochs  3000 \
--t5-path  /storage/zhubin/LlamaGen/dataset/storage_datasets_npy \
--gpt-type t2v \
--dataset t2v \
--num-workers 8  \
--log-every 1  \
--ckpt-every  3000  \
--results-dir results_vae_1f_shuffle \
--gpt-ckpt /storage/zhubin/LlamaGen/results_vae_1f_mask/000-GPT-B/checkpoints/0005000.pt \
--save_train_video_latent 
 
