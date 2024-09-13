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


import sys
current_file_path = os.path.abspath(__file__)
current_directory = os.path.dirname(current_file_path)
sys.path.append(os.path.join(current_directory, '..'))

from utils.distributed import init_distributed_mode
from language.t5 import T5Embedder

# CAPTION_KEY = {
#     'blip': 0,
#     'llava': 1,
#     'llava_first': 2,
# }
#################################################################################
#                             Training Helper Functions                         #
#################################################################################
class CustomDataset(Dataset):
    def __init__(self, file_path, trunc_nums=1000):
        meta_info = []

        if file_path.endswith('.jsonl'):
            with open(file_path, 'r') as file:
                for line_idx, line in enumerate(file):
                    data = json.loads(line)
                    meta_info.append(data)
        elif file_path.endswith('.json'):
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)[:trunc_nums]
            # import ipdb; ipdb.set_trace()
            for item in data:
                video_rela_path = item['path']
                dir_name = os.path.dirname(video_rela_path)
                filename = os.path.splitext(os.path.basename(video_rela_path))[0]
                caption = item['cap'][-1]
                meta_info.append([caption, dir_name, filename])


        self.meta_info = meta_info

    def __len__(self):
        return len(self.meta_info)

    def __getitem__(self, index):
        caption, code_dir, code_name = self.meta_info[index] # caption, dir_name, filename
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
    dataset = CustomDataset(args.file_path, args.trunc_nums)
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

    for caption, code_dir, code_name in loader:
        
        caption_embs, emb_masks = t5_xxl.get_text_embeddings(caption)
        valid_caption_embs = caption_embs[:, :emb_masks.sum()]
        x = valid_caption_embs.to(torch.float32).detach().cpu().numpy()
        os.makedirs(os.path.join(args.t5_path, code_dir[0]), exist_ok=True)
        np.save(os.path.join(args.t5_path, code_dir[0], '{}.npy'.format(code_name[0])), x)
        print(code_name[0])

    dist.destroy_process_group()


def tmp():
    import json, random   
    # 读取原始JSON文件
    with open('/storage/zhubin/video_statistics_data/task1.5/Final_format_dataset_data_v2/step1.5_istock_final_815070.json', 'r') as file:
        data = json.load(file)

    # 随机抽取3000个元素
    random_data = json.loads(json.dumps(data))
    random.shuffle(random_data)
    random_data = random_data[:3000]

    # 将新的列表保存到新的JSON文件中
    with open('/storage/zhubin/video_statistics_data/task1.5/Final_format_dataset_data_v2/step1.5_istock_final_815070_random_3000.json', 'w') as file:
        json.dump(random_data, file, indent=2)


if __name__ == "__main__":
    # tmp()
    parser = argparse.ArgumentParser()
    # parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--file_path", type=str, required=True)
    parser.add_argument("--t5-path", type=str, required=True)
    parser.add_argument("--trunc_nums", type=int, required=True)
    parser.add_argument("--num_workers", type=int, default=0)
    # parser.add_argument("--caption-key", type=str, default='blip', choices=list(CAPTION_KEY.keys()))
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
    export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

  
    torchrun --nnodes=1 --nproc_per_node=1 \
    --master_port=29503 \
    language/extract_t5_feature_custom_v2.py \
    --file_path  /storage/zhubin/video_statistics_data/task1.5/Final_format_dataset_data_v2/step1.5_istock_final_815070_random_3000.json \
    --t5-path /storage/zhubin/LlamaGen/dataset/storage_datasets_npy \
    --t5-model-path   /storage/zhubin/LlamaGen/pretrained_models/t5-ckpt/ \
    --t5-model-type flan-t5-xl \
    --trunc_nums 3000 \
    --num_workers 0
    

    # ===flowers
    cd /storage/zhubin/LlamaGen/
    conda activate motionctrl
    export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
 
    torchrun --nnodes=1 --nproc_per_node=8 \
    --master_port=29503 \
    language/extract_t5_feature_custom_v2.py \
    --file_path  /storage/zhubin/LlamaGen/dataset/Image_Datasets/flowers/meta_data.json \
    --t5-path /storage/zhubin/LlamaGen/dataset/storage_datasets_npy/flowers \
    --t5-model-path   /storage/zhubin/LlamaGen/pretrained_models/t5-ckpt/ \
    --t5-model-type flan-t5-xl \
    --trunc_nums 40000 \
    --num_workers 0
    """