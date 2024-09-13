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

from autoregressive.models.gpt_video import GPT_models
from autoregressive.models.generate_video import generate

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
    
    assert os.path.exists(args.t5_path)
    t5_model = T5Embedder(
        device=device, 
        local_cache=True, 
        cache_dir=args.t5_path, 
        dir_or_name=args.t5_model_type,
        torch_dtype=precision,
        model_max_length=args.t5_feature_max_len,
    )
    prompts = [
        "The video captures a serene view of a city skyline with tall buildings and a bridge stretching across a body of water. The scene is bathed in soft light, creating a tranquil atmosphere. The water surface is dotted with small ripples and reflections of the cityscape. The sky above is filled with light, possibly from the sun setting or rising, casting a warm glow over the entire scene. The overall mood of the video is peaceful and calming, inviting viewers to appreciate the beauty of the urban landscape.",
        "The video presents a static aerial view of a cityscape bisected by a river, under an overcast sky, creating a serene nocturnal or pre-dawn atmosphere. The skyline is densely packed with buildings of varying heights, illuminated by artificial lighting. A bridge spans the river, its lights reflecting on the water's surface, enhancing the mirror-like effect of the calm water. Throughout the video, the scene remains largely unchanged, maintaining the tranquil mood with the vertical streaks of light from the buildings and bridge consistently overlaying the cityscape. The lighting conditions, camera angle, and the overall composition of the scene stay the same, indicating no significant passage of time or change in the visual elements of the city by the river under the cloudy sky.",
        "The video captures a black and white dog as it traverses a snowy landscape, moving consistently from left to right across the frame. Initially, the dog is seen with its head lowered, possibly sniffing or interacting with the snow-covered ground, its path marked by a series of paw prints indicating recent activity. The backdrop features a dense forest of tall, bare trees, suggesting a wintry setting. As the dog continues its journey, there is a moment where it raises its head, possibly changing its pace or focus, before lowering it again, presumably to continue investigating the snowy terrain. At one point, the dog pauses, standing still with its head raised, attentive to its surroundings. The natural lighting throughout casts soft shadows on the snow, highlighting the dog's movements and the texture of the landscape. Throughout this sequence, the dog's proximity to the camera varies slightly, indicating either movement away from or towards the lens, with the surrounding environment remaining consistent.",
        "In the serene setting of a snowy forest, a black and white dog, possibly a Border Collie, is captured in a moment of tranquility. The dog, with its coat as dark as the night and as pure as the snow, is walking on a path blanketed in a thick layer of snow. The dog's head is lowered, perhaps sniffing the ground or simply enjoying the crisp winter air. The forest around it is a winter wonderland, with trees standing tall and silent, their branches heavy with snow. The ground is a vast expanse of white, untouched except for the dog's footprints. The scene is peaceful and quiet, with only the sound of the dog's footsteps breaking the stillness of the snowy landscape."
         "The video depicts a person working meticulously on a white washing machine, initially seen using a screwdriver to adjust or repair a component near the machine's top part. The washing machine's lid is open, revealing the inner workings, and the person is dressed in a grey shirt and blue gloves, indicating a setting that likely involves repair or maintenance. The background throughout is plain and light-colored, emphasizing the task at hand without providing specific details about the location. Over time, the person's actions remain focused on the washing machine, with slight adjustments in the positioning of their hands and arms as they continue their work with the screwdriver. The environment and the person's attire do not change, and there is no significant camera movement, ensuring the viewer's attention stays on the repair activity. The video captures the person's concentrated effort and attention to detail as they carry out the repair or maintenance task on the washing machine.",
         "In the video, a person is meticulously working on the intricate components of a machine, which appears to be a modern appliance or a piece of equipment. The individual is wearing protective gloves and a uniform that suggests a professional setting, possibly a repair technician or an engineer. The machine is open, revealing its internal workings, and the person's hands are focused on connecting or adjusting wires and components. The scene is well-lit, highlighting the precision and care taken in the task at hand."
        "The video features an individual from a low-angle perspective, giving a sense of being observed from below. The person is indoors, with a background that hints at a library or a room filled with bookshelves, although the details are blurred. Throughout the video, the individual's hand is consistently raised to their head, suggesting a gesture of thought or contemplation. There is no significant change in the environment, the individual's actions, or the camera's perspective, which remains focused on the person, maintaining the initial sense of observation or scrutiny.",
        "In the video, a young woman is engrossed in a conversation on her cell phone. She is standing in a room filled with bookshelves, suggesting a library or a personal study area. The woman is dressed in casual attire and appears to be deeply engaged in her phone call. Her expression and body language indicate that the conversation is of importance or interest to her. The room around her is well-lit, with natural light streaming in, casting a warm glow on the scene. The bookshelves in the background are filled with a variety of books, hinting at a love for literature or a scholarly pursuit."
         "The video depicts a person in a kitchen setting, engaged in the process of gift-wrapping, focusing on the hands as they work with a pink ribbon on a white gift box. Initially, the person is seen tying a knot on the box, with scissors lying on the counter, indicating that the wrapping process is underway. As the video progresses, the hands are observed adjusting and tightening the ribbon around the gift box, with movements suggesting the final touches to the wrapping are being made. The actions include pulling the ribbon to adjust the bow and fine-tuning the placement of the knot. Throughout the video, the background remains consistent, featuring a kitchen counter and a white backsplash, with the camera maintaining a close-up shot to emphasize the wrapping activity. The scissors remain unmoved on the counter, and the focus is kept on the hands and the gift box, concluding with the hands ensuring the ribbon is neatly secured on the box.",
       "In the video, a person's hands are meticulously tying a vibrant pink ribbon around a white box. The hands move with precision, ensuring the ribbon is tied evenly and neatly. The person's fingers deftly manipulate the ribbon, creating a bow at the top of the box. The bow is then carefully adjusted to ensure it sits perfectly on the box. The hands repeat this process, tying another ribbon around the box, creating a symmetrical effect. The video showcases the artistry and care involved in gift wrapping, with the hands taking great care to make the presentation look as appealing as possible."
         "The video features a spider as the central subject against a blurred background, providing a detailed and close-up view of the arthropod throughout its duration. The spider is depicted with a prominent display of its legs, which are covered in a mix of brown and black hairs, and its body showcases a shiny texture due to the lighting that highlights its form. The focus on the spider's texture and form is maintained across the video, with no noticeable movement from the spider itself, indicating that it remains stationary. The background remains consistently out of focus, suggesting an indoor setting with artificial lighting. Throughout the video, there are no significant changes in the spider's position, the camera angle, or the lighting conditions, indicating a static scene where the spider is the sole focus, and no action or environmental changes occur.",
      "The video captures a close-up view of a spider's intricate web of long, thin, and delicate legs. The spider's body is not visible, but its web is the main focus of the shot. The camera remains stationary, allowing the viewer to observe the spider's web in detail. The spider's web is a complex structure, with each strand meticulously placed, showcasing the spider's remarkable ability to create such a delicate and strong network. The video provides a unique perspective on the spider's world, highlighting the beauty and complexity of its web."
    
        # "A blue Porsche 356 parked in front of a yellow brick wall.",
        # "A photo of an astronaut riding a horse in the forest. There is a river in front of them with water lilies.",
        # "A map of the United States made out of sushi. It is on a table next to a glass of red wine."
    ]

    """caption_embs, emb_masks = t5_model.get_text_embeddings(prompts)

    if not args.no_left_padding:
        print(f"processing left-padding...")    
        # a naive way to implement left-padding
        new_emb_masks = torch.flip(emb_masks, dims=[-1])
        new_caption_embs = []
        for idx, (caption_emb, emb_mask) in enumerate(zip(caption_embs, emb_masks)):
            valid_num = int(emb_mask.sum().item())
            print(f'  prompt {idx} token len: {valid_num}')
            new_caption_emb = torch.cat([caption_emb[valid_num:], caption_emb[:valid_num]])
            new_caption_embs.append(new_caption_emb)
        new_caption_embs = torch.stack(new_caption_embs)
    else:
        new_caption_embs, new_emb_masks = caption_embs, emb_masks
    c_indices = new_caption_embs * new_emb_masks[:,:, None]
    c_emb_masks = new_emb_masks

    import ipdb; ipdb.set_trace()"""

    t5_file = '/storage/zhubin/LlamaGen/dataset/storage_datasets_npy/istock/videos_istock_coco/a-young-woman-looking-up-at-the-sky-gm1370107313-439725734.npy'
    t5_file = '/storage/zhubin/LlamaGen/dataset/storage_datasets_npy/istock/videos_istock_coco/manga-or-comic-book-lines-animation-action-speed-effects-with-clouds-sun-rays-gm1299945277-392407375.npy'
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

    python3 autoregressive/sample/sample_t2v_1f.py  \
        --vae-model  VAE-16 \
        --vae-ckpt ${CKPT} \
        --vq-ckpt ./pretrained_models/vq_ds16_t2i.pt \
        --gpt-ckpt /storage/zhubin/LlamaGen/results_vae_1f_mask/000-GPT-B/checkpoints/00010000.pt  \
        --t5-model-path  pretrained_models/t5-ckpt \
        --t5-model-type  flan-t5-xl \
        --downsample-size 32 \
        --image-size 256 \
        --gpt-type t2v \
        --t5-path  /storage/zhubin/LlamaGen/pretrained_models/t5-ckpt/  \
        --gpt-model GPT-B     \
        --cfg-scale 1 \
        --num_frames 1 






        

    """