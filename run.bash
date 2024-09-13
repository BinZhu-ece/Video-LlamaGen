# !/bin/bash


#================ 第一步先提取  图片的caption T5特征 ================

export CUDA_VISIBLE_DEVICES=7
# /storage/zhubin/LlamaGen/language/extract_t5_feature_custom.py

# ===================== 测试casual videovae
conda activate vae
bash /storage/zhubin/LlamaGen/CausalVideoVAE/scripts/causalvideovae_gen_video.sh

# 新的vae

bash /storage/zhubin/LlamaGen/CausalVideoVAE/scripts/causalvideovae_gen_video_hw_squeeze_1s.sh

"""
/storage/zhubin/LlamaGen/CausalVideoVAE/scripts/rec_causalvideo_vae.py :


vqvae = CausalVAEModel.from_pretrained(args.ckpt)
if args.enable_tiling:
    vqvae.enable_tiling()
    vqvae.tile_overlap_factor = args.tile_overlap_factor
vqvae = vqvae.to(rank).to(data_type)
vqvae.eval()


x, file_names = batch['video'], batch['file_name']
x = x.to(device=device, dtype=data_type)  # b c t h w
latents = vqvae.encode(x).sample().to(data_type)
video_recon = vqvae.decode(latents)

"""
# 训练的时候要加上 --enable_tiling, 推理高分辨长时长视频就得开不然爆显存
# --change_decoder 需要吗？
#================ 第二步训练  Video-VAE ================




# ======== t2i ===========

export master_addr=127.0.0.1
export master_port=29506
export CUDA_VISIBLE_DEVICES=6
 
cd /storage/zhubin/LlamaGen 
conda activate motionctrl
torchrun --nnodes=1  --nproc_per_node=1 --master_addr=$master_addr --master_port=$master_port \
autoregressive/train/train_t2i.py \
--vq-ckpt ./pretrained_models/vq_ds16_t2i.pt \
--data-path /storage/zhubin/LlamaGen/dataset/Image_Datasets/  \
--t5-feat-path /storage/zhubin/LlamaGen/dataset/Image_Datasets/  \
--dataset t2i \
--image-size 256 \
--cloud-save-path ./cloud_path  \
--global-batch-size 2 \
--log-every 10 \
--epochs 300 \
--ckpt-every 1000 

# --log-every 
# --num-workers 24


# ======== t2v dataset ==========
source  /storage/miniconda3/etc/profile.d/conda.sh 
conda activate motionctrl
bash /storage/zhubin/LlamaGen/scripts/dataset/t2v_dataset.bash


# ======== t2v =========
export CUDA_VISIBLE_DEVICES=7
export master_addr=127.0.0.1
export master_port=29502
export CUDA_VISIBLE_DEVICES=7
conda activate motionctrl
torchrun \
--nnodes=1 --nproc_per_node=1  \
--master_addr=$master_addr --master_port=$master_port \
autoregressive/train/train_t2v.py \
--vae-ckpt /storage/zhubin/LlamaGen/CausalVideoVAE/vae_ckpt  \
--enable_tiling \
--data-path /storage/zhubin/LlamaGen/dataset/Image_Datasets/  \
--t5-feat-path /storage/zhubin/LlamaGen/dataset/Image_Datasets/  \
--dataset t2i \
--image-size 256 \
--cloud-save-path ./cloud_path  \
--global-batch-size 256 \
--log-every 10 \
--epochs 300 \
--ckpt-every 1000 


"""

推理：
export http_proxy=127.0.01:7895
export https_proxy=127.0.01:7895
export CUDA_VISIBLE_DEVICES=6
conda activate motionctrl
# 官方推理
cd  /storage/zhubin/LlamaGen/
python3 autoregressive/sample/sample_t2i.py --vq-ckpt ./pretrained_models/vq_ds16_t2i.pt --gpt-ckpt ./pretrained_models/t2i_XL_stage1_256.pt --gpt-model GPT-XL --image-size 256
# 复现权重推理
python3 autoregressive/sample/sample_t2i.py --vq-ckpt ./pretrained_models/vq_ds16_t2i.pt --gpt-ckpt  ./results/021-GPT-B/checkpoints/0011000.pt  --gpt-model GPT-B --image-size 256




"""
