CKPT=/storage/zhubin/LlamaGen/CausalVideoVAE/vae_ckpt/488dim8
nnodes=1
nproc_per_node=1
export master_addr=127.0.0.1
export master_port=29505
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
cd /storage/zhubin/LlamaGen
source  /storage/miniconda3/etc/profile.d/conda.sh 
conda activate motionctrl

python3 autoregressive/sample/sample_t2v_1f_diff.py  \
    --vae-model  VAE-16 \
    --vae-ckpt ${CKPT} \
    --vq-ckpt ./pretrained_models/vq_ds16_t2i.pt \
    --t5-model-path  pretrained_models/t5-ckpt \
    --t5-model-type  flan-t5-xl \
    --downsample-size 8 \
    --image-size 256 \
    --gpt-type t2v \
    --t5-path  /storage/zhubin/LlamaGen/pretrained_models/t5-ckpt/  \
    --gpt-model GPT-1B     \
    --cfg-scale 1 \
    --num_frames 1 \
    --gpt-ckpt /storage/zhubin/LlamaGen/results_vae_1f_diff_flowers_1B/009-GPT-1B/checkpoints/0040000.pt  \
    --sample_t5_dir  /storage/zhubin/LlamaGen/dataset/storage_datasets_npy/flowers/100 \
    --precision none 

    # --gpt-ckpt /storage/zhubin/LlamaGen/results_vae_1f_mask/000-GPT-B/checkpoints/0009000.pt  \
    # --sample_t5_dir  /storage/zhubin/LlamaGen/dataset/storage_datasets_npy/istock/videos_istock_coco