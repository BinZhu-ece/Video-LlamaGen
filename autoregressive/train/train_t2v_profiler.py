# Modified from:
#   fast-DiT: https://github.com/chuanyangjin/fast-DiT
#   nanoGPT: https://github.com/karpathy/nanoGPT
# LLamaGen: https://github.com/FoundationVision/LlamaGen

import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms
from glob import glob
import time
import argparse
import os
from tqdm import tqdm 

import sys
# 获取当前文件的目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# 向上回溯两级找到所需的目录
parent_dir = os.path.dirname(os.path.dirname(current_dir))
# 将找到的目录添加到sys.path
sys.path.append(parent_dir)

from utils.distributed import init_distributed_mode
from utils.logger import create_logger
from dataset.build import build_dataset
from dataset.augmentation import center_crop_arr
# from autoregressive.train.train_c2i import creat_optimizer
# from autoregressive.models.gpt import GPT_models
# from autoregressive.models.gpt_video import GPT_models
from autoregressive.models.gpt_video_mask_diff import GPT_models

from tokenizer.tokenizer_image.vq_model import VQ_models
from tokenizer.tokenizer_image.vae_model import VAE_models
sys.path.append(os.path.join('./','CausalVideoVAE'))

from language.t5 import T5Embedder
from dataset.transform import ToTensorVideo, TemporalRandomCrop, RandomHorizontalFlipVideo, CenterCropResizeVideo, LongSideResizeVideo, SpatialStrideCropVideo
from torchvision.transforms import Lambda

import numpy as np
import cv2
import numpy.typing as npt
import inspect

#################################################################################
#                             Training Helper Functions                         #
#################################################################################
def creat_optimizer(model, weight_decay, learning_rate, betas, logger):
    # start with all of the candidate parameters
    param_dict = {pn: p for pn, p in model.named_parameters()}
    # filter out those that do not require grad
    param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
    # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
    # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
    decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
    nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
    optim_groups = [
        {'params': decay_params, 'weight_decay': weight_decay},
        {'params': nodecay_params, 'weight_decay': 0.0}
    ]
    num_decay_params = sum(p.numel() for p in decay_params)
    num_nodecay_params = sum(p.numel() for p in nodecay_params)
    logger.info(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
    logger.info(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
    # Create AdamW optimizer and use the fused version if it is available
    fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
    extra_args = dict(fused=True) if fused_available else dict()
    optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
    logger.info(f"using fused AdamW: {fused_available}")
    return optimizer


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
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."
    
    # Setup DDP:
    init_distributed_mode(args)
    assert args.global_batch_size % dist.get_world_size() == 0, f"Batch size must be divisible by world size."
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)

    # Setup an experiment folder:
    if rank == 0:
        os.makedirs(args.results_dir, exist_ok=True)  # Make results folder (holds all experiment subfolders)
        experiment_index = len(glob(f"{args.results_dir}/*"))
        model_string_name = args.gpt_model.replace("/", "-") 
        experiment_dir = f"{args.results_dir}/{experiment_index:03d}-{model_string_name}"
        checkpoint_dir = f"{experiment_dir}/checkpoints"
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(experiment_dir)
        logger.info(f"Experiment directory created at {experiment_dir}")

        time_record = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
        cloud_results_dir = f"{args.cloud_save_path}/{time_record}"
        cloud_checkpoint_dir = f"{cloud_results_dir}/{experiment_index:03d}-{model_string_name}/checkpoints"
        os.makedirs(cloud_checkpoint_dir, exist_ok=True)
        logger.info(f"Experiment directory created in cloud at {cloud_checkpoint_dir}")
    
    else:
        logger = create_logger(None)

    # training args
    logger.info(f"{args}")

    # training env
    logger.info(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")


    #  VAE setup
    if args.dataset == 't2i':     # create and load model
        vq_model = VQ_models[args.vq_model](
            codebook_size=args.codebook_size,
            codebook_embed_dim=args.codebook_embed_dim)
        vq_model.to(device)
        vq_model.eval()
        checkpoint = torch.load(args.vq_ckpt, map_location="cpu")
        vq_model.load_state_dict(checkpoint["model"])
        del checkpoint   
    elif args.dataset == 't2v':
        # args-- vae_model+enable_tiling+vae_ckpt   CausalVAEModel.from_pretrained(args.vae_ckpt)
        CausalVAEModel = VAE_models[args.vae_model] 
        vae = CausalVAEModel.from_pretrained(args.vae_ckpt)
        if args.enable_tiling:
            vae.enable_tiling()
            vae.tile_overlap_factor = args.tile_overlap_factor
        vae = vae.to(device) # .to(data_type)
        vae.eval()

    # Setup model
    latent_size = args.image_size // args.downsample_size
    model = GPT_models[args.gpt_model](
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
    ).to(device)
    logger.info(f"GPT Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Setup optimizer
    optimizer = creat_optimizer(model, args.weight_decay, args.lr, (args.beta1, args.beta2), logger)

    # T5 model
    assert os.path.exists(args.t5_model_path)
    precision = {'none': torch.float32, 'bf16': torch.bfloat16, 'fp16': torch.float16}[args.mixed_precision]
    t5_xxl = T5Embedder(
        device=device, 
        local_cache=True, 
        cache_dir=args.t5_model_path, 
        dir_or_name=args.t5_model_type,
        torch_dtype=precision
    )

    # transform 
    resize = [CenterCropResizeVideo((args.max_height, args.max_width)), ]
    norm_fun = Lambda(lambda x: 2. * x - 1.)
    transform = transforms.Compose([
        ToTensorVideo(),
        *resize, 
        # RandomHorizontalFlipVideo(p=0.5),  # in case their caption have position decription
        norm_fun
    ])

    # dataset
    dataset = build_dataset(args, transform=transform, t5_xxl = t5_xxl)
    sampler = DistributedSampler(
        dataset,
        num_replicas=dist.get_world_size(),
        rank=rank,
        shuffle=True,
        seed=args.global_seed
    )
    loader = DataLoader(
        dataset,
        batch_size=int(args.global_batch_size // dist.get_world_size()),
        shuffle=False,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        # prefetch_factor=args.prefetch_factor,
    )
    logger.info(f"Dataset contains {len(dataset):,} images")

    # Prepare models for training:
    if args.gpt_ckpt:
        checkpoint = torch.load(args.gpt_ckpt, map_location="cpu")
        model.load_state_dict(checkpoint["model"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        train_steps = checkpoint["steps"] if "steps" in checkpoint else int(args.gpt_ckpt.split('/')[-1].split('.')[0])
        start_epoch = int(train_steps / int(len(dataset) / args.global_batch_size))
        train_steps = int(start_epoch * int(len(dataset) / args.global_batch_size))
        del checkpoint
        logger.info(f"Resume training from checkpoint: {args.gpt_ckpt}")
        logger.info(f"Initial state: steps={train_steps}, epochs={start_epoch}")
    else:
        train_steps = 0
        start_epoch = 0

    if not args.no_compile:
        logger.info("compiling the model... (may take several minutes)")
        model = torch.compile(model) # requires PyTorch 2.0        
    
    model = DDP(model.to(device), device_ids=[args.gpu],  find_unused_parameters=True)
    model.train()  # important! This enables embedding dropout for classifier-free guidance

    # initialize a GradScaler. If enabled=False scaler is a no-op
    scaler = torch.cuda.amp.GradScaler(enabled=(args.mixed_precision =='fp16'))
    # Variables for monitoring/logging purposes:
    log_steps = 0
    running_loss = 0
    start_time = time.time()

    logger.info(f"Training for {args.epochs} epochs...")
    for epoch in range(start_epoch, args.epochs):
        sampler.set_epoch(epoch)
        logger.info(f"Beginning epoch {epoch}...")  
        # import ipdb; ipdb.set_trace()

        with torch.profiler.profile(
                schedule=torch.profiler.schedule(wait=1, warmup=1, active=1, repeat=1),
                on_trace_ready=torch.profiler.tensorboard_trace_handler('./log/llamagen'),
                record_shapes=True,
                profile_memory=True,
                with_stack=True
        ) as prof:
            step = 0
            for sample in  loader:

                prof.step()  # Need to call this at each step to notify profiler of steps' boundary.
                if step >= 10:
                    break
                step += 1

                x, y, attn_mask, valid = sample['video_data']['video'], sample['video_data']['t5_feat_padding'], sample['video_data']['attn_mask'], sample['video_data']['valid']
                x = x.to(device, non_blocking=True) # torch.Size([2, 3, 17, 256, 256])
                y = y.to(device, non_blocking=True) # torch.Size([2, 1, 512])
                attn_mask = attn_mask.to(device, non_blocking=True)
                valid = valid.to(device, non_blocking=True)
                """
                x: torch.Size([2, 3, 17, 256, 256])
                y: torch.Size([2, 1, 120, 2048])
                attn_mask: torch.Size([2, 1, 1400, 1400])
                valid: torch.Size([2])
                """
                if args.dataset == 't2i':
                    img = x
                    with torch.no_grad():
                        _, _, [_, _, indices] = vq_model.encode(img)
                    x = indices.reshape(img.shape[0], -1)
                elif args.dataset == 't2v':
                    video = x
                    with torch.no_grad():
                        z = vae.encode(video).sample() # vae.encode(video): DiagonalGaussianDistribution  after sample: torch.Size([2, 2048, 5, 16, 16]) (bs, dim, t, h, w)
                    video_latent =  z.flatten(2).transpose(1, 2) # (b, c, t, h, w)  ->  (b, t*h*w, c)
                    c_indices = y.reshape(y.shape[0], y.shape[-2], y.shape[-1]) # no change
                    cond_embed = c_indices

                
                attn_mask = attn_mask.reshape(attn_mask.shape[0], 1, attn_mask.shape[-2], attn_mask.shape[-1]) # (bs, n_head, seq_len, seq_len)
                
                # import ipdb; ipdb.set_trace()
                if args.save_train_video_latent:
                    logger.info(f"!!!save_train_video_latent...")  
                    with torch.cuda.amp.autocast(dtype=precision):  
                        _, loss, pred_vae_latent  = model(
                                idx = None,
                                cond_idx=None,
                                targets=None,
                                # cond_idx=c_indices,
                                # idx=z_indices[:, :-1],
                                # targets=z_indices,
                                mask=attn_mask[:, :, :-1, :-1], # torch.Size([2, 1, 1400, 1400])
                                valid=valid, # torch.Size([2])
                                video_latent=video_latent[:, :-1], # torch.Size([2, 1280, 2048])
                                targets_video=video_latent,
                                cond_embed=cond_embed, # torch.Size([2, 120, 2048])
                                save_train_video_latent=args.save_train_video_latent,
                                )
                else:
                    _, loss  = model(
                                idx = None,
                                cond_idx=None,
                                targets=None,
                                # cond_idx=c_indices,
                                # idx=z_indices[:, :-1],
                                # targets=z_indices,
                                mask=attn_mask[:, :, :-1, :-1], # torch.Size([2, 1, 1400, 1400])
                                valid=valid, # torch.Size([2])
                                video_latent=video_latent, #  [:, :-1], # torch.Size([2, 1280, 2048])
                                targets_video=video_latent,
                                cond_embed=cond_embed, # torch.Size([2, 120, 2048])
                                )
                # import ipdb; ipdb.set_trace()
                if args.save_train_video_latent:
                    logger.info(f"args.save_train_video_latent222222222")  
                    if rank == 0:
                        import ipdb; ipdb.set_trace()
                        # pred_vae_latent = None
                        #     num_frames = args.num_frames,   t_downsample_size = args.t_downsample_size, 
                        token_embedding_sample = pred_vae_latent.view(-1, (args.num_frames-1)//args.t_downsample_size + 1, latent_size, latent_size, pred_vae_latent.shape[-1])
                        token_embedding_sample = token_embedding_sample.permute(0, 4, 1, 2, 3)
                        vae = vae.to(token_embedding_sample.dtype)
                        with torch.no_grad():
                            samples = vae.decode(token_embedding_sample)
                        for idx, video in enumerate(samples):
                            custom_to_video(
                                video, fps=16, output_file=f'{idx}.mp4'
                            )

                        token_embedding_gt = video_latent.view(-1, (args.num_frames-1)//args.t_downsample_size + 1, latent_size, latent_size, pred_vae_latent.shape[-1])
                        token_embedding_gt = token_embedding_gt.permute(0, 4, 1, 2, 3)
                        vae = vae.to(token_embedding_gt.dtype)
                        with torch.no_grad():
                            samples_gt = vae.decode(token_embedding_gt)
                        for idx, video in enumerate(samples_gt):
                            custom_to_video(
                                video, fps=16, output_file=f'{idx}_gt.mp4'
                            )
                # import ipdb; ipdb.set_trace()
                # backward pass, with gradient scaling if training in fp16         
                scaler.scale(loss).backward()
                if args.max_grad_norm != 0.0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                # step the optimizer and scaler if training in fp16
                scaler.step(optimizer)
                scaler.update()
                # flush the gradients as soon as we can, no need for this memory anymore
                optimizer.zero_grad(set_to_none=True)

                # Log loss values:
                running_loss += loss.item()
                log_steps += 1
                train_steps += 1
                if train_steps % args.log_every == 0:
                    # Measure training speed:
                    torch.cuda.synchronize()
                    end_time = time.time()
                    steps_per_sec = log_steps / (end_time - start_time)
                    # Reduce loss history over all processes:
                    avg_loss = torch.tensor(running_loss / log_steps, device=device)
                    dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                    avg_loss = avg_loss.item() / dist.get_world_size()
                    logger.info(f"(step={train_steps:07d}) Train Loss: {avg_loss:.4f}, Train Steps/Sec: {steps_per_sec:.2f}")
                    # Reset monitoring variables:
                    running_loss = 0
                    log_steps = 0
                    start_time = time.time()

                # Save checkpoint:
                if train_steps % args.ckpt_every == 0 and train_steps > 0:
                    logger.info(f"Start saving checkpoint!")
                    if rank == 0:
                        if not args.no_compile:
                            model_weight = model.module._orig_mod.state_dict()
                        else:
                            model_weight = model.module.state_dict()  
                        checkpoint = {
                            "model": model_weight,
                            "optimizer": optimizer.state_dict(),
                            "steps": train_steps,
                            "args": args
                        }
                        if not args.no_local_save:
                            checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
                            torch.save(checkpoint, checkpoint_path)
                            logger.info(f"Saved checkpoint to {checkpoint_path}")
                        
                        cloud_checkpoint_path = f"{cloud_checkpoint_dir}/{train_steps:07d}.pt"
                        torch.save(checkpoint, cloud_checkpoint_path)
                        logger.info(f"Saved checkpoint in cloud to {cloud_checkpoint_path}")
                    dist.barrier()



    model.eval()  # important! This disables randomized embedding dropout
    # do any sampling/FID calculation/etc. with ema (or model) in eval mode ...

    logger.info("Done!")
    dist.destroy_process_group()


import traceback
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, required=True)
    # parser.add_argument("--t5-feat-path", type=str, required=True)
    parser.add_argument("--short-t5-feat-path", type=str, default=None, help="short caption of t5_feat_path")
    parser.add_argument("--cloud-save-path", type=str, required=True, help='please specify a cloud disk path, if not, local path')
    parser.add_argument("--no-local-save", action='store_true', help='no save checkpoints to local path for limited disk volume')
    parser.add_argument("--vq-model", type=str, choices=list(VQ_models.keys()), default="VQ-16")
    parser.add_argument("--vq-ckpt", type=str, default=None, help="ckpt path for vq model")
    parser.add_argument("--codebook-size", type=int, default=16384, help="codebook size for vector quantization")
    parser.add_argument("--codebook-embed-dim", type=int, default=8, help="codebook dimension for vector quantization")
    parser.add_argument("--gpt-model", type=str, choices=list(GPT_models.keys()), default="GPT-B")
    parser.add_argument("--gpt-ckpt", type=str, default=None, help="ckpt path for resume training")
    parser.add_argument("--gpt-type", type=str, choices=['c2i', 't2i', 't2v'], default="t2v")
    parser.add_argument("--vocab-size", type=int, default=16384, help="vocabulary size of visual tokenizer")
    parser.add_argument("--cls-token-num", type=int, default=120, help="max token number of condition input")
    parser.add_argument("--dropout-p", type=float, default=0.1, help="dropout_p of resid_dropout_p and ffn_dropout_p")
    parser.add_argument("--token-dropout-p", type=float, default=0.1, help="dropout_p of token_dropout_p")
    parser.add_argument("--drop-path", type=float, default=0.0, help="drop_path_rate of attention and ffn")
    parser.add_argument("--no-compile", action='store_true')
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--dataset", type=str, default='t2i')
    parser.add_argument("--image-size", type=int, choices=[256, 384, 512], default=384)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=5e-2, help="Weight decay to use.")
    parser.add_argument("--beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--beta2", type=float, default=0.95, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--max-grad-norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--global-batch-size", type=int, default=256)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=24)
    parser.add_argument("--prefetch_factor", type=int, default=8)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--ckpt-every", type=int, default=5000)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--mixed-precision", type=str, default='bf16', choices=["none", "fp16", "bf16"])

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
    # parser.add_argument("--precision", type=str, default='bf16')
    parser.add_argument("--model_max_length", type=int, default=512)
    parser.add_argument("--start_frame_ind", type=int, default=25)
    parser.add_argument("--num_frames", type=int, default=17)
    parser.add_argument("--data_root", type=str, default='/storage/dataset')

    parser.add_argument("--t-downsample-size", type=int, choices=[4, 8], default=4)
    parser.add_argument("--downsample-size", type=int, choices=[8, 16], default=16)

    parser.add_argument("--t5-path", type=str, required=True)
    parser.add_argument('--save_train_video_latent', action='store_true')
    
    args = parser.parse_args()
    try:
        main(args)
    except Exception as e:
        traceback.print_exc()

    """
    
    # print(x.shape, y.shape, attn_mask.shape, valid.shape)
    # continue
    
    # for epoch in range(start_epoch, args.epochs):
    #     sampler.set_epoch(epoch)
    #     logger.info(f"Beginning epoch {epoch}...")  
    #     # import ipdb; ipdb.set_trace()
    #     for i in  range(1000000):
    #         batchsize = int(args.global_batch_size // dist.get_world_size())
    #         x = torch.randn(batchsize, 3, 17, 256, 256)
    #         y = torch.randn(batchsize, 1, 120, 2048)
    #         attn_mask = torch.ones(batchsize, 1, 1400, 1400)
    #         valid = torch.ones(batchsize)
    
    """