import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision('high')
setattr(torch.nn.Linear, 'reset_parameters', lambda self: None)     # disable default parameter init for faster speed
setattr(torch.nn.LayerNorm, 'reset_parameters', lambda self: None)  # disable default parameter init for faster speed
from torchvision.utils import save_image

import os
import time
import argparse


import sys
# 获取当前文件的目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# 向上回溯两级找到所需的目录
parent_dir = os.path.dirname(os.path.dirname(current_dir))
# 将找到的目录添加到sys.path
sys.path.append(parent_dir)


from tokenizer.tokenizer_image.vae_model import VAE_models

from language.t5 import T5Embedder

# from autoregressive.models.gpt_video import GPT_models
from autoregressive.models.generate_video_diff import generate
from autoregressive.models.gpt_video_diff import GPT_models
os.environ["TOKENIZERS_PARALLELISM"] = "false"


import numpy as np
import cv2
import numpy.typing as npt

def array_to_video(
    image_array: npt.NDArray, fps: float = 30.0, output_file: str = "output_video.mp4"
) -> None:
    height, width, channels = image_array[0].shape
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(output_file, fourcc, float(fps), (width, height))
    for image in image_array:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        video_writer.write(image_rgb)

    video_writer.release()

def custom_to_video(
    x: torch.Tensor, fps: float = 2.0, output_file: str = "output_video.mp4"
) -> None:
    x = x.detach().cpu()
    x = torch.clamp(x, -1, 1)
    x = (x + 1) / 2
    x = x.permute(1, 2, 3, 0).float().numpy()
    x = (255 * x).astype(np.uint8)
    array_to_video(x, fps=fps, output_file=output_file)
    return


def main(args):

    # Setup PyTorch:
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # create and load model
    # args-- vae_model+enable_tiling+vae_ckpt   CausalVAEModel.from_pretrained(args.vae_ckpt)
    CausalVAEModel = VAE_models[args.vae_model] 
    vae = CausalVAEModel.from_pretrained(args.vae_ckpt)
    if args.enable_tiling:
        vae.enable_tiling()
        vae.tile_overlap_factor = args.tile_overlap_factor
    vae = vae.to(device) # .to(data_type)
    vae.eval()
    print(f"video vae is loaded")

    # create and load gpt model
    precision = {'none': torch.float32, 'bf16': torch.bfloat16, 'fp16': torch.float16}[args.precision]
    latent_size = args.image_size // args.downsample_size

    
    gpt_model = GPT_models[args.gpt_model](
        vocab_size=args.vocab_size,
        block_size=latent_size ** 2,
        num_classes=args.num_classes,
        cls_token_num=args.cls_token_num,
        model_type=args.gpt_type,
        resid_dropout_p=args.dropout_p,
        ffn_dropout_p=args.dropout_p,
        token_dropout_p=args.token_dropout_p,
        vae_embed_dim =  vae.config.embed_dim, # add vae latent dim
        num_frames = args.num_frames,
        t_downsample_size = args.t_downsample_size, 
    ).to(device, dtype=precision)

    checkpoint = torch.load(args.gpt_ckpt, map_location="cpu")
 
    if "model" in checkpoint:  # ddp
        model_weight = checkpoint["model"]
    elif "module" in checkpoint: # deepspeed
        model_weight = checkpoint["module"]
    elif "state_dict" in checkpoint:
        model_weight = checkpoint["state_dict"]
    else:
        raise Exception("please check model weight")
    gpt_model.load_state_dict(model_weight, strict=False)
    gpt_model.eval()
    del checkpoint
    print(f"gpt model is loaded")

    if args.compile:
        print(f"compiling the model...")
        gpt_model = torch.compile(
            gpt_model,
            mode="reduce-overhead",
            fullgraph=True
        ) # requires PyTorch 2.0 (optional)
    else:
        print(f"no need to compile model in demo") 
    
    # assert os.path.exists(args.t5_path)
    # t5_model = T5Embedder(
    #     device=device, 
    #     local_cache=True, 
    #     cache_dir=args.t5_path, 
    #     dir_or_name=args.t5_model_type,
    #     torch_dtype=precision,
    #     model_max_length=args.t5_feature_max_len,
    # )
    # t5_file = '/storage/zhubin/LlamaGen/dataset/storage_datasets_npy/istock/videos_istock_coco/a-young-woman-looking-up-at-the-sky-gm1370107313-439725734.npy'
    # t5_file = '/storage/zhubin/LlamaGen/dataset/storage_datasets_npy/istock/videos_istock_coco/manga-or-comic-book-lines-animation-action-speed-effects-with-clouds-sun-rays-gm1299945277-392407375.npy'
    import glob
    t5_files = glob.glob(f'{args.sample_t5_dir}/*.npy')
    
    for IDX, t5_file in enumerate(t5_files):
        assert os.path.isfile(t5_file), 't5_file {} does not exist!'.format(t5_file)
        t5_feat = torch.from_numpy(np.load(t5_file))
        t5_feat_len = t5_feat.shape[1] 
        feat_len = min(120, t5_feat_len)
        emb_mask = torch.zeros((120,))
        emb_mask[-feat_len:] = 1
        emb_mask = emb_mask.unsqueeze(0)
        # import ipdb; ipdb.set_trace()
        t5_feat_padding = torch.zeros((1, 120, t5_feat.shape[-1]))
        t5_feat_padding[:, -feat_len:] = t5_feat[:, :feat_len]
        c_indices = t5_feat_padding*emb_mask[:,:, None]
        c_emb_masks = emb_mask
        c_indices = c_indices.to(device, dtype=precision)
        c_emb_masks = c_emb_masks.to(device, dtype=precision)


        # qzshape = [len(c_indices), args.codebook_embed_dim, latent_size, latent_size]
        t1 = time.time()
        # import ipdb; ipdb.set_trace()
        vae_t = (args.num_frames-1)//args.t_downsample_size+1
        token_embedding_sample = generate(
            gpt_model, c_indices, vae_t * (latent_size ** 2), 
            c_emb_masks, 
            cfg_scale=args.cfg_scale,
            temperature=args.temperature, top_k=args.top_k,
            top_p=args.top_p, sample_logits=True, 
            )
        
        # import ipdb; ipdb.set_trace()
        sampling_time = time.time() - t1
        print(f"Full sampling takes about {sampling_time:.2f} seconds.")    
        token_embedding_sample = token_embedding_sample.view(-1, vae_t, latent_size, latent_size, token_embedding_sample.shape[-1])
        token_embedding_sample = token_embedding_sample.permute(0, 4, 1, 2, 3)

        t2 = time.time()
        # samples = vq_model.decode_code(index_sample, qzshape) # output value is between [-1, 1]
        vae = vae.to(token_embedding_sample.dtype)
        samples = vae.decode(token_embedding_sample)
        decoder_time = time.time() - t2
        print(f"decoder takes about {decoder_time:.2f} seconds.")

        
        for idx, video in enumerate(samples):
            custom_to_video(
                video, fps=16, output_file=f'gen-{IDX}-{idx}.mp4'
            )

        # save_image(samples, "sample_{}.mp4".format(args.gpt_type), nrow=4, normalize=True, value_range=(-1, 1))
        # print(f"image is saved to sample_{args.gpt_type}.png")


import traceback
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("--t5-path", type=str, default='pretrained_models/t5-ckpt')
    # parser.add_argument("--t5-model-type", type=str, default='flan-t5-xl')
    parser.add_argument("--t5-feature-max-len", type=int, default=120)
    parser.add_argument("--t5-feature-dim", type=int, default=2048)
    parser.add_argument("--no-left-padding", action='store_true', default=False)
    parser.add_argument("--gpt-model", type=str, choices=list(GPT_models.keys()), default="GPT-XL")
    parser.add_argument("--gpt-ckpt", type=str, default='/storage/zhubin/LlamaGen/results/108-GPT-B/checkpoints/0001500.pt')
    parser.add_argument("--gpt-type", type=str, choices=['c2i', 't2i', 't2v'], default="t2i", help="class->image or text->image")  
    parser.add_argument("--cls-token-num", type=int, default=120, help="max token number of condition input")
    # parser.add_argument("--precision", type=str, default='bf16', choices=["none", "fp16", "bf16"]) 
    parser.add_argument("--vocab-size", type=int, default=16384, help="vocabulary size of visual tokenizer")
    parser.add_argument("--dropout-p", type=float, default=0.1, help="dropout_p of resid_dropout_p and ffn_dropout_p")
    parser.add_argument("--token-dropout-p", type=float, default=0.1, help="dropout_p of token_dropout_p")
    parser.add_argument("--drop-path", type=float, default=0.0, help="drop_path_rate of attention and ffn")
    
    parser.add_argument("--compile", action='store_true', default=False)
    parser.add_argument("--vq-model", type=str, choices=list(VAE_models.keys()), default="VAE-16")
    parser.add_argument("--vq-ckpt", type=str, default=None, help="ckpt path for vq model")
    parser.add_argument("--codebook-size", type=int, default=16384, help="codebook size for vector quantization")
    parser.add_argument("--codebook-embed-dim", type=int, default=8, help="codebook dimension for vector quantization")
    parser.add_argument("--image-size", type=int, choices=[256, 384, 512], default=512)
    
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--cfg-scale", type=float, default=7.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=1000, help="top-k value to sample with")
    parser.add_argument("--temperature", type=float, default=1.0, help="temperature value to sample with")
    parser.add_argument("--top-p", type=float, default=1.0, help="top-p value to sample with")

    # vae
    parser.add_argument("--vae-model", type=str, choices=list(VAE_models.keys()), default="VAE-16")
    parser.add_argument("--vae-ckpt", type=str, default='/storage/zhubin/LlamaGen/CausalVideoVAE/vae_ckpt', help="ckpt path for vq model")
    parser.add_argument('--enable_tiling', action='store_true')
    parser.add_argument("--tile_overlap_factor", type=float, default=0.25)

    # dataset related
    parser.add_argument("--video_meta_info_file", type=str, default='/storage/zhubin/liuyihang/add_aes/output/sucai_aes_1000.json')
    parser.add_argument("--t5-model-path", type=str, default='./pretrained_models/t5-ckpt')
    parser.add_argument("--t5-model-type", type=str, default='flan-t5-xl')
    parser.add_argument("--max_height", type=int, default=256)
    parser.add_argument("--max_width", type=int, default=256)
    parser.add_argument("--precision", type=str, default='bf16')
    parser.add_argument("--model_max_length", type=int, default=512)
    parser.add_argument("--start_frame_ind", type=int, default=25)
    parser.add_argument("--num_frames", type=int, default=17)

    parser.add_argument("--t5-path", type=str, required=True)
    parser.add_argument("--downsample-size", type=int, choices=[8, 16, 32], default=32)
    parser.add_argument("--t-downsample-size", type=int, choices=[4, 8], default=4)
    parser.add_argument("--sample_t5_dir", type=str, default='/storage/zhubin/LlamaGen/dataset/storage_datasets_npy/istock/videos_istock_coco')
    
    # parser.add_argument("--precision", type=str, default='bf16')
    # 
    args = parser.parse_args()
    try:
        main(args)
    except Exception as e:
        traceback.print_exc()

    """
    CKPT=/storage/zhubin/LlamaGen/CausalVideoVAE/vae_ckpt/432322048
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
        --gpt-ckpt /storage/zhubin/LlamaGen/results_vae_1f_mask_diff_flowers_1B/003-GPT-1B/checkpoints/0016000.pt  \
        --t5-model-path  pretrained_models/t5-ckpt \
        --t5-model-type  flan-t5-xl \
        --downsample-size 32 \
        --image-size 256 \
        --gpt-type t2v \
        --t5-path  /storage/zhubin/LlamaGen/pretrained_models/t5-ckpt/  \
        --gpt-model GPT-1B     \
        --cfg-scale 1 \
        --num_frames 1 \
        --precision none \
        --sample_t5_dir /storage/zhubin/LlamaGen/dataset/storage_datasets_npy/flowers/0






        

    """