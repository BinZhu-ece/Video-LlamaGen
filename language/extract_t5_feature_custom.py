import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import numpy as np
import argparse
import os
import json

import sys; sys.path.append('/storage/zhubin/LlamaGen/')
from utils.distributed import init_distributed_mode
from language.t5 import T5Embedder

CAPTION_KEY = {
    'blip': 0,
    'llava': 1,
    'llava_first': 2,
}
#################################################################################
#                             Training Helper Functions                         #
#################################################################################
class CustomDataset(Dataset):
    def __init__(self, file_path):
        img_path_list = []

        # t5_file = os.path.join(self.t5_feat_path, code_dir, f"{code_name}.npy")
        with open(file_path, 'r') as file:
            for line_idx, line in enumerate(file):
                data = json.loads(line)
                img_path_list.append(data)
        self.img_path_list = img_path_list
        

    def __len__(self):
        return len(self.img_path_list)

    def __getitem__(self, index):
        (caption, code_dir, code_name) = self.img_path_list[index]
        return caption, code_dir, code_name


        
#################################################################################
#                                  Training Loop                                #
#################################################################################
def main(args):
    """
    Trains a new DiT model.
    """
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."

    # Setup DDP:
    # dist.init_process_group("nccl")
    init_distributed_mode(args)
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")

    # Setup a feature folder:
    if rank == 0:
        os.makedirs(args.t5_path, exist_ok=True)

    # Setup data:
    print(f"Dataset is preparing...")
    dataset = CustomDataset(args.file_path)
    sampler = DistributedSampler(
        dataset,
        num_replicas=dist.get_world_size(),
        rank=rank,
        shuffle=False,
        seed=args.global_seed
    )
    loader = DataLoader(
        dataset,
        batch_size=1, # important!
        shuffle=False,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False
    )
    print(f"Dataset contains {len(dataset):,} images")

    precision = {'none': torch.float32, 'bf16': torch.bfloat16, 'fp16': torch.float16}[args.precision]
    assert os.path.exists(args.t5_model_path)
    # import ipdb; ipdb.set_trace()
    t5_xxl = T5Embedder(
        device=device, 
        local_cache=True, 
        cache_dir=args.t5_model_path, 
        dir_or_name=args.t5_model_type,
        torch_dtype=precision
    )

    from tqdm import tqdm
    for caption, code_dir, code_name in tqdm(loader):

        # import ipdb; ipdb.set_trace()
        caption_embs, emb_masks = t5_xxl.get_text_embeddings(caption)
        valid_caption_embs = caption_embs[:, :emb_masks.sum()]
        x = valid_caption_embs.to(torch.float32).detach().cpu().numpy()
        os.makedirs(os.path.join(args.t5_path, code_dir[0]), exist_ok=True)

        save_file = os.path.join(args.t5_path, code_dir[0], '{}.npy'.format(code_name[0]))
        np.save(save_file, x)
        print(save_file)

    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_path", type=str, required=True)
    parser.add_argument("--t5-path", type=str, required=True)
    # parser.add_argument("--data-start", type=int, required=True)
    # parser.add_argument("--data-end", type=int, required=True)
    parser.add_argument("--caption-key", type=str, default='blip', choices=list(CAPTION_KEY.keys()))
    parser.add_argument("--trunc-caption", action='store_true', default=False)
    parser.add_argument("--t5-model-path", type=str, default='./pretrained_models/t5-ckpt')
    parser.add_argument("--t5-model-type", type=str, default='flan-t5-xl')
    parser.add_argument("--precision", type=str, default='bf16', choices=["none", "fp16", "bf16"])
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=24)
    args = parser.parse_args()
    main(args)

    """
    cd /storage/zhubin/LlamaGen/
    conda activate motionctrl
    export CUDA_VISIBLE_DEVICES=7

    torchrun --nnodes=1 --nproc_per_node=1 --node_rank=0 \
    --master_port=29501 \
    language/extract_t5_feature_custom.py \
    --file_path  /storage/zhubin/LlamaGen/dataset/Image_Datasets/civitai_v1_10000.jsonl  \
    --t5-path /storage/zhubin/LlamaGen/dataset/Image_Datasets/civitai_v1_1940032_flan_t5_xl \
    --t5-model-path   /storage/zhubin/LlamaGen/pretrained_models \
    --t5-model-type flan-t5-xl \
    --caption-key blip 
    
    """