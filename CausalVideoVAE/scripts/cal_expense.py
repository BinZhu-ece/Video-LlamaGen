import sys
sys.path.append(".")
from causalvideovae.model import CausalVAEModel
from CV_VAE.models.modeling_vae import CVVAEModel
from opensora.models.vae.vae import VideoAutoencoderPipeline
from opensora.registry import DATASETS, MODELS, build_module
from opensora.utils.config_utils import parse_configs
from diffusers.models import AutoencoderKL, AutoencoderKLTemporalDecoder
from tats import VQGAN
from tats.download import load_vqgan
from taming.models.vqgan import VQModel, GumbelVQ
import torch
from omegaconf import OmegaConf
import yaml
import argparse
from einops import rearrange
from causalvideovae.model.modules.normalize import Normalize
from causalvideovae.model.modules.block import Block
import time

def total_params(model):
    total_params = sum(p.numel() for p in model.parameters())
    total_params_in_millions = total_params / 1e6
    return total_params_in_millions
  
device = torch.device('cuda')
data_type = torch.bfloat16
video_input = torch.randn(1, 3, 33, 256, 256).to(device).to(data_type)
image_input = torch.randn(33, 3, 256, 256).to(device).to(data_type)
num = 1000

"""
#VQGAN
def load_config(config_path, display=False):
  config = OmegaConf.load(config_path)
  if display:
    print(yaml.dump(OmegaConf.to_container(config)))
  return config

def load_vqgan(config, ckpt_path=None, is_gumbel=False):
  if is_gumbel:
    model = GumbelVQ(**config.model.params)
  else:
    model = VQModel(**config.model.params)
  if ckpt_path is not None:
    sd = torch.load(ckpt_path, map_location="cpu")["state_dict"]
    missing, unexpected = model.load_state_dict(sd, strict=False)
  return model.eval()

vqgan_ckpt='/remote-home1/clh/taming-transformers/logs/vqgan_gumbel_f8/checkpoints/last.ckpt'
vqgan_config='/remote-home1/clh/taming-transformers/logs/vqgan_gumbel_f8/configs/model.yaml'
vqgan_config = load_config(vqgan_config, display=False)
vqgan = load_vqgan(vqgan_config, ckpt_path=vqgan_ckpt, is_gumbel=True).to(device).to(data_type).eval()
vqgan.requires_grad_(False)
print('VQGAN')
print(f"Generator:\t\t{total_params(vqgan) :.2f}M")
print(f"\t- Encoder:\t{total_params(vqgan.encoder) :.2f}M")
print(f"\t- Decoder:\t{total_params(vqgan.decoder) :.2f}M")
# 计算程序运行时间  
start_time = time.time()
for i in range(num):
  latents, _, [_, _, indices] = vqgan.encode(image_input)
end_time = time.time()  
print(f"encode_time:{(end_time - start_time)/num :.3f}s")

start_time = time.time()
for i in range(num):  
  video_recon = vqgan.decode(latents.to(data_type))
end_time = time.time()  
print(f"decode_time:{(end_time - start_time)/num :.3f}s")

start_time = time.time()
for i in range(num):
  latents, _, [_, _, indices] = vqgan.encode(image_input)
  video_recon = vqgan.decode(latents.to(data_type))
end_time = time.time()  
print(f"rec_time:{(end_time - start_time)/num :.3f}s")



#TATS
tats_path = '/remote-home1/clh/TATS/vqgan_sky_128_488_epoch_12-step_29999-train.ckpt'
tats = VQGAN.load_from_checkpoint(tats_path).to(device).to(torch.float32).eval()
tats.requires_grad_(False)
print('TATS')
print(f"Generator:\t\t{total_params(tats) :.2f}M")
print(f"\t- Encoder:\t{total_params(tats.encoder):.2f}M")
print(f"\t- Decoder:\t{total_params(tats.decoder):.2f}M")
# 计算程序运行时间  
start_time = time.time()
for i in range(num):
  z = tats.pre_vq_conv(tats.encoder(video_input.to(torch.float32)))
  vq_output = tats.codebook(z)
  latents = vq_output['embeddings']
end_time = time.time()  
print(f"encode_time:{(end_time - start_time)/num :.3f}s")

start_time = time.time()
for i in range(num):  
  video_recon = tats.decoder(tats.post_vq_conv(latents.to(torch.float32)))
end_time = time.time()  
print(f"decode_time:{(end_time - start_time)/num :.3f}s")

start_time = time.time()
for i in range(num):
  z = tats.pre_vq_conv(tats.encoder(video_input.to(torch.float32)))
  vq_output = tats.codebook(z)
  latents = vq_output['embeddings']
  video_recon = tats.decoder(tats.post_vq_conv(latents.to(torch.float32)))
end_time = time.time()  
print(f"rec_time:{(end_time - start_time)/num :.3f}s")


#SD2_1
sd2_1_path = '/remote-home1/clh/sd2_1'
sd2_1 = AutoencoderKL.from_pretrained(sd2_1_path).eval().to(device).to(data_type)
sd2_1.requires_grad_(False)
print('SD2_1')
print(f"Generator:\t\t{total_params(sd2_1) :.2f}M")
print(f"\t- Encoder:\t{total_params(sd2_1.encoder):.2f}M")
print(f"\t- Decoder:\t{total_params(sd2_1.decoder):.2f}M")
# 计算程序运行时间  
start_time = time.time()
for i in range(num):
  latents = sd2_1.encode(image_input)['latent_dist'].sample()
end_time = time.time()  
print(f"encode_time:{(end_time - start_time)/num :.3f}s")

start_time = time.time()
for i in range(num):  
  video_recon = sd2_1.decode(latents.to(data_type))['sample']
end_time = time.time()  
print(f"decode_time:{(end_time - start_time)/num :.3f}s")

start_time = time.time()
for i in range(num):
  latents = sd2_1.encode(image_input)['latent_dist'].sample()
  video_recon = sd2_1.decode(latents.to(data_type))['sample']
end_time = time.time()  
print(f"rec_time:{(end_time - start_time)/num :.3f}s")

#SVD
svd_path = '/remote-home1/clh/svd/'
svd = AutoencoderKLTemporalDecoder.from_pretrained(svd_path).eval().to(device).to(data_type)
svd.requires_grad_(False)
print('SVD')
print(f"Generator:\t\t{total_params(svd):.2f}M")
print(f"\t- Encoder:\t{total_params(svd.encoder):.2f}M")
print(f"\t- Decoder:\t{total_params(svd.decoder):.2f}M")
# 计算程序运行时间  
start_time = time.time()
for i in range(num):
  latents = svd.encode(image_input)['latent_dist'].sample()
end_time = time.time()  
print(f"encode_time:{(end_time - start_time)/num :.3f}s")

start_time = time.time()
for i in range(num):  
  video_recon = svd.decode(latents.to(data_type), num_frames=video_input.shape[2])['sample']
end_time = time.time()  
print(f"decode_time:{(end_time - start_time)/num :.3f}s")

start_time = time.time()
for i in range(num):
  latents = svd.encode(image_input)['latent_dist'].sample()
  video_recon = svd.decode(latents.to(data_type), num_frames=video_input.shape[2])['sample']
end_time = time.time()  
print(f"rec_time:{(end_time - start_time)/num :.3f}s")

#CV-VAE
cvvae_path = '/remote-home1/clh/CV-VAE/vae3d'
cvvae = CVVAEModel.from_pretrained(cvvae_path).eval().to(device).to(data_type)
cvvae.requires_grad_(False)
print('CV-VAE')
print(f"Generator:\t\t{total_params(cvvae):.2f}M")
print(f"\t- Encoder:\t{total_params(cvvae.encoder):.2f}M")
print(f"\t- Decoder:\t{total_params(cvvae.decoder):.2f}M")
# 计算程序运行时间  
start_time = time.time()
for i in range(num):
  latent = cvvae.encode(video_input).latent_dist.sample()
end_time = time.time()  
print(f"encode_time:{(end_time - start_time)/num :.3f}s")

start_time = time.time()
for i in range(num):  
 video_recon = cvvae.decode(latent).sample
end_time = time.time()  
print(f"decode_time:{(end_time - start_time)/num :.3f}s")

start_time = time.time()
for i in range(num):
  latent = cvvae.encode(video_input).latent_dist.sample()
  video_recon = cvvae.decode(latent).sample
end_time = time.time()  
print(f"rec_time:{(end_time - start_time)/num :.3f}s")


#NUS-VAE
nusvae_path = '/remote-home1/clh/CV-VAE/vae3d'
parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, default="/remote-home1/clh/Causal-Video-VAE/opensora/video.py")
parser.add_argument("--ckpt", type=str, default="/remote-home1/clh/Open-Sora/OpenSora-VAE-v1.2")
args = parser.parse_args()
cfg = parse_configs(args, training=False)
nusvae = build_module(cfg.model, MODELS).eval().to(device).to(data_type)
nusvae.requires_grad_(False)
print('NUS-VAE')
print(f"Generator:\t\t{total_params(nusvae):.2f}M")
print(f"\t- Spatial_Encoder:\t{total_params(nusvae.spatial_vae.module.encoder):.2f}M")
print(f"\t- Temporal_Encoder:\t{total_params(nusvae.temporal_vae.encoder):.2f}M")
print(f"\t- Temporal_Decoder:\t{total_params(nusvae.temporal_vae.decoder):.2f}M")
print(f"\t- Spatial_Decoder:\t{total_params(nusvae.spatial_vae.module.decoder):.2f}M")
# 计算程序运行时间  
start_time = time.time()
for i in range(num):
  latents, posterior, x_z = nusvae.encode(video_input)
end_time = time.time()  
print(f"encode_time:{(end_time - start_time)/num :.3f}s")

start_time = time.time()
for i in range(num):  
 video_recon, x_z_rec = nusvae.decode(latents, num_frames=video_input.size(2))
end_time = time.time()  
print(f"decode_time:{(end_time - start_time)/num :.3f}s")

start_time = time.time()
for i in range(num):
  latents, posterior, x_z = nusvae.encode(video_input)
  video_recon, x_z_rec = nusvae.decode(latents, num_frames=video_input.size(2))
end_time = time.time()  
print(f"rec_time:{(end_time - start_time)/num :.3f}s")
"""

#ours1.2
ours1_2_vae_path = '/remote-home1/clh/models/23denc_3ddec_vae_pretrained_weight'
ours1_2_vae = CausalVAEModel.from_pretrained(ours1_2_vae_path).eval().to(device).to(data_type)
ours1_2_vae.requires_grad_(False)
print('open_sora_plan_vae_1_2')
print(f"Generator:\t\t{total_params(ours1_2_vae):.2f}M")
print(f"\t- Encoder:\t{total_params(ours1_2_vae.encoder):.2f}M")
print(f"\t- Decoder:\t{total_params(ours1_2_vae.decoder):.2f}M")
# 计算程序运行时间  
start_time = time.time()
for i in range(num):
  latents = ours1_2_vae.encode(video_input).sample().to(data_type)
end_time = time.time()  
print(f"encode_time:{(end_time - start_time)/num :.3f}s")

start_time = time.time()
for i in range(num):  
 video_recon = ours1_2_vae.decode(latents)
end_time = time.time()  
print(f"decode_time:{(end_time - start_time)/num :.3f}s")

start_time = time.time()
for i in range(num):
  latents = ours1_2_vae.encode(video_input).sample().to(data_type)
  video_recon = ours1_2_vae.decode(latents)
end_time = time.time()  
print(f"rec_time:{(end_time - start_time)/num :.3f}s")


#23d
half3d_vae_path = '/remote-home1/clh/models/23d_vae_pretrained_weight'
half3d_vae = CausalVAEModel.from_pretrained(half3d_vae_path).eval().to(device).to(data_type)
half3d_vae.requires_grad_(False)
print('open_sora_plan_vae_half3d')
print(f"Generator:\t\t{total_params(half3d_vae):.2f}M")
print(f"\t- Encoder:\t{total_params(half3d_vae.encoder):.2f}M")
print(f"\t- Decoder:\t{total_params(half3d_vae.decoder):.2f}M")
# 计算程序运行时间  
start_time = time.time()
for i in range(num):
  latents = half3d_vae.encode(video_input).sample().to(data_type)
end_time = time.time()  
print(f"encode_time:{(end_time - start_time)/num :.3f}s")

start_time = time.time()
for i in range(num):  
 video_recon = half3d_vae.decode(latents)
end_time = time.time()  
print(f"decode_time:{(end_time - start_time)/num :.3f}s")

start_time = time.time()
for i in range(num):
  latents = half3d_vae.encode(video_input).sample().to(data_type)
  video_recon = half3d_vae.decode(latents)
end_time = time.time()  
print(f"rec_time:{(end_time - start_time)/num :.3f}s")

#2and3d
mix23d_vae_path = '/remote-home1/clh/models/mix23d_vae_pretrained_weight'
mix23d_vae = CausalVAEModel.from_pretrained(mix23d_vae_path).eval().to(device).to(data_type)
mix23d_vae.requires_grad_(False)
print('open_sora_plan_vae_mix23d')
print(f"Generator:\t\t{total_params(mix23d_vae):.2f}M")
print(f"\t- Encoder:\t{total_params(mix23d_vae.encoder):.2f}M")
print(f"\t- Decoder:\t{total_params(mix23d_vae.decoder):.2f}M")
# 计算程序运行时间  
start_time = time.time()
for i in range(num):
  latents = mix23d_vae.encode(video_input).sample().to(data_type)
end_time = time.time()  
print(f"encode_time:{(end_time - start_time)/num :.3f}s")

start_time = time.time()
for i in range(num):  
 video_recon = mix23d_vae.decode(latents)
end_time = time.time()  
print(f"decode_time:{(end_time - start_time)/num :.3f}s")

start_time = time.time()
for i in range(num):
  latents = mix23d_vae.encode(video_input).sample().to(data_type)
  video_recon = mix23d_vae.decode(latents)
end_time = time.time()  
print(f"rec_time:{(end_time - start_time)/num :.3f}s")

#full 3d
full3d_vae_path = '/remote-home1/clh/models/full3d_vae_pretrained_weight'
full3d_vae = CausalVAEModel.from_pretrained(full3d_vae_path).eval().to(device).to(data_type)
full3d_vae.requires_grad_(False)
print('open_sora_plan_vae_full3d')
print(f"Generator:\t\t{total_params(full3d_vae):.2f}M")
print(f"\t- Encoder:\t{total_params(full3d_vae.encoder):.2f}M")
print(f"\t- Decoder:\t{total_params(full3d_vae.decoder):.2f}M")
# 计算程序运行时间  
start_time = time.time()
for i in range(num):
  latents = full3d_vae.encode(video_input).sample().to(data_type)
end_time = time.time()  
print(f"encode_time:{(end_time - start_time)/num :.3f}s")

start_time = time.time()
for i in range(num):  
 video_recon = full3d_vae.decode(latents)
end_time = time.time()  
print(f"decode_time:{(end_time - start_time)/num :.3f}s")

start_time = time.time()
for i in range(num):
  latents = full3d_vae.encode(video_input).sample().to(data_type)
  video_recon = full3d_vae.decode(latents)
end_time = time.time()  
print(f"rec_time:{(end_time - start_time)/num :.3f}s")

