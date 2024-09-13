# NCCL setting
export GLOO_SOCKET_IFNAME=bond0
export NCCL_SOCKET_IFNAME=bond0
export NCCL_IB_HCA=mlx5_10:1,mlx5_11:1,mlx5_12:1,mlx5_13:1
export NCCL_IB_GID_INDEX=3
export NCCL_IB_TC=162
export NCCL_IB_TIMEOUT=22
export NCCL_PXN_DISABLE=0
export NCCL_IB_QPS_PER_CONNECTION=4
# export NCCL_ALGO=Ring
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NCCL_ALGO=Tree

cd /storage/zhubin/LlamaGen 

REAL_DATASET_DIR=/storage/clh/gen/288/train/origin
EXP_NAME=tmp1
SAMPLE_RATE=1
NUM_FRAMES=65
RESOLUTION=720
SUBSET_SIZE=100
CKPT=/storage/lcm/Causal-Video-VAE/results/288_resume-lr1.00e-05-bs1-rs256-sr2-fr25/77
OUTPUT_VIDEO_DIR=useless

source /storage/miniconda3/etc/profile.d/conda.sh 
conda activate /storage/miniconda3/envs/vae
export CUDA_VISIBLE_DEVICES=6
torchrun \
     --nnodes=1 --nproc_per_node=1 \
    --rdzv_endpoint=localhost:29504 \
    --master_addr=localhost \
    --master_port=29600 \
    CausalVideoVAE/scripts/rec_causalvideo_vae.py \
    --batch_size 1 \
    --real_video_dir ${REAL_DATASET_DIR} \
    --generated_video_dir $OUTPUT_VIDEO_DIR \
    --device cuda \
    --sample_fps 24 \
    --sample_rate ${SAMPLE_RATE} \
    --num_frames ${NUM_FRAMES} \
    --resolution ${RESOLUTION} \
    --crop_size ${RESOLUTION} \
    --num_workers 8 \
    --ckpt ${CKPT} \
    --output_origin \
    --enable_tiling \
    #--change_decoder \
    #--decoder_dir /remote-home1/clh/Causal-Video-VAE/results/decoder_only-lr1.00e-05-bs1-rs248-sr2-fr25/checkpoint-5000.ckpt \