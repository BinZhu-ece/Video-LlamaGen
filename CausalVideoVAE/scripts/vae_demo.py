import random
import argparse
import cv2
from tqdm import tqdm
import numpy as np
import numpy.typing as npt
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, Subset
from decord import VideoReader, cpu
from torch.nn import functional as F
from pytorchvideo.transforms import ShortSideScale
from torchvision.transforms import Lambda, Compose
from torchvision.transforms._transforms_video import CenterCropVideo
import sys
from torch.utils.data import Dataset, DataLoader, Subset
import os
import glob
sys.path.append(".")
from causalvideovae.model import CausalVAEModel
import gradio as gr
from functools import partial  
 
ckpt = 0
device = 0    
data_type = 0
rank = 0


def ddp_setup():
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

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

def _format_video_shape(video, time_compress=4, spatial_compress=8):
    time = video.shape[1]
    height = video.shape[2]
    width = video.shape[3]
    new_time = (
        (time - (time - 1) % time_compress)
        if (time - 1) % time_compress != 0
        else time
    )
    new_height = (
        (height - (height) % spatial_compress)
        if height % spatial_compress != 0
        else height
    )
    new_width = (
        (width - (width) % spatial_compress) if width % spatial_compress != 0 else width
    )
    return video[:, :new_time, :new_height, :new_width]

def read_video(video_path: str, num_frames: int, sample_rate: int) -> torch.Tensor:
    decord_vr = VideoReader(video_path, ctx=cpu(0), num_threads=8)
    total_frames = len(decord_vr)
    sample_frames_len = sample_rate * num_frames

    if total_frames > sample_frames_len:
        s = 0
        e = s + sample_frames_len
        num_frames = num_frames
    else:
        s = 0
        e = total_frames
        num_frames = int(total_frames / sample_frames_len * num_frames)
        print(
            f"sample_frames_len {sample_frames_len}, only can sample {num_frames * sample_rate}",
            video_path,
            total_frames,
        )

    frame_id_list = np.linspace(s, e - 1, num_frames, dtype=int)
    video_data = decord_vr.get_batch(frame_id_list).asnumpy()
    video_data = torch.from_numpy(video_data)
    video_data = video_data.permute(3, 0, 1, 2)  # (T, H, W, C) -> (C, T, H, W)
    return video_data




   
        
        

    





@torch.no_grad()
def vae_rec(input_file):
    
    our_vae_path = '/storage/clh/Causal-Video-VAE/gradio/temp/video.mp4'
    if input_file.endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp')):  
        # 处理图片  
        print("处理图片...")  
        # 这里添加您的图片处理代码  
        return "图片处理结果"  # 假设这是处理后的图片或相关信息  
    elif input_file.endswith(('.mp4', '.avi', '.mov', '.wmv')):  
        # 处理视频
        decord_vr = VideoReader(input_file, ctx=cpu(0))
        total_frames = len(decord_vr)
        # 加载视频  
        video = cv2.VideoCapture(input_file)  
        # 获取FPS  
        fps = video.get(cv2.CAP_PROP_FPS)
        frame_id_list = np.linspace(0, total_frames-1, total_frames, dtype=int)
        video_data = decord_vr.get_batch(frame_id_list).asnumpy()
        video_data = torch.from_numpy(video_data)
        video_data = video_data.permute(3, 0, 1, 2)
        video_data = (video_data / 255.0) * 2 - 1
        video_data = _format_video_shape(video_data)
        video_data = video_data.unsqueeze(0)
        video_data = video_data.to(device=device, dtype=data_type)  # b c t h w
        
        ##我们的VAE的输出
        latents = vqvae.encode(video_data).sample().to(data_type)
        video_recon = vqvae.decode(latents)
        custom_to_video(video_recon[0], fps=fps, output_file=our_vae_path)
        ##
        
        return  input_file, our_vae_path
    else:  
        return "不支持的文件类型"
    
    

    
@torch.no_grad()
def main(args: argparse.Namespace):
  
    # 创建输入界面  
    input_interface = gr.components.File(label="上传文件（图片或视频）")  
  
    # 创建输出界面  
    output_video1 = gr.Video(label="原始视频和图片")  
    output_video2 = gr.Video(label="我们的3D VAE输出视频和图片") 

    iface = gr.Interface(fn=vae_rec,  
                     inputs=input_interface,  
                     outputs=[output_video1, output_video2], 
                     title="Open-Sora-Plan 3D Causal VAE",  
                     description="这是简单的3D VAE展示界面，用户可以上传视频或者图像，我们会对其进行重建。\
                     This is a simple 3D VAE demonstration interface. Users can upload videos or images, and we will reconstruct them.",
                     examples=['/storage/clh/Causal-Video-VAE/gradio/reconstruct_video/134445.mp4'])  
  
    # 启动Gradio应用  
    iface.launch()
    

   

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--real_video_dir", type=str, default="")
    parser.add_argument("--generated_video_dir", type=str, default="")
    parser.add_argument("--decoder_dir", type=str, default="")
    parser.add_argument("--ckpt", type=str, default="")
    parser.add_argument("--sample_fps", type=int, default=30)
    parser.add_argument("--resolution", type=int, default=336)
    parser.add_argument("--crop_size", type=int, default=None)
    parser.add_argument("--num_frames", type=int, default=17)
    parser.add_argument("--sample_rate", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--subset_size", type=int, default=None)
    parser.add_argument("--tile_overlap_factor", type=float, default=0.25)
    parser.add_argument('--enable_tiling', action='store_true')
    parser.add_argument('--output_origin', action='store_true')
    parser.add_argument('--change_decoder', action='store_true')
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()
    device = args.device    
    data_type = torch.bfloat16
    ddp_setup()
    rank = int(os.environ["LOCAL_RANK"])
    vqvae = CausalVAEModel.from_pretrained(args.ckpt)
    if args.enable_tiling:
        vqvae.enable_tiling()
        vqvae.tile_overlap_factor = args.tile_overlap_factor
    vqvae = vqvae.to(rank).to(data_type)
    vqvae.eval()
    main(args)