import os
import json
import numpy as np

import torch
from torch.utils.data import Dataset
from PIL import Image 
import sys
current_file_path = os.path.abspath(__file__)
current_directory = os.path.dirname(current_file_path)
sys.path.append(os.path.join(current_directory, '..'))
from dataset.augmentation import center_crop_arr

class Text2ImgDatasetImg(Dataset):
    def __init__(self, lst_dir, face_lst_dir, transform):
        img_path_list = []
        valid_file_path = []
        # collect valid jsonl
        for lst_name in sorted(os.listdir(lst_dir)):
            if not lst_name.endswith('.jsonl'):
                continue
            file_path = os.path.join(lst_dir, lst_name)
            valid_file_path.append(file_path)
        
        # collect valid jsonl for face
        if face_lst_dir is not None:
            for lst_name in sorted(os.listdir(face_lst_dir)):
                if not lst_name.endswith('_face.jsonl'):
                    continue
                file_path = os.path.join(face_lst_dir, lst_name)
                valid_file_path.append(file_path)            
        
        for file_path in valid_file_path:
            with open(file_path, 'r') as file:
                for line_idx, line in enumerate(file):
                    data = json.loads(line)
                    img_path = data['image_path']
                    code_dir = file_path.split('/')[-1].split('.')[0]
                    img_path_list.append((img_path, code_dir, line_idx))
        self.img_path_list = img_path_list
        self.transform = transform

    def __len__(self):
        return len(self.img_path_list)

    def __getitem__(self, index):
        img_path, code_dir, code_name = self.img_path_list[index]
        img = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, code_name 


class Text2ImgDataset(Dataset):
    def __init__(self, args, transform):
        img_path_list = []
        valid_file_path = []
        # collect valid jsonl file path
        for lst_name in sorted(os.listdir(args.data_path)):
            if not lst_name.endswith('.jsonl'):
                continue
            file_path = os.path.join(args.data_path, lst_name)
            valid_file_path.append(file_path)           
        
        for file_path in valid_file_path:
            with open(file_path, 'r') as file:
                for line_idx, line in enumerate(file):
                    data = json.loads(line)
                    img_path = data['image_path']
                    code_dir = file_path.split('/')[-1].split('.')[0]
                    # img_path_list.append((img_path, code_dir, line_idx))
                    # the code_name of our dataset is not by line_idx
                    code_name = img_path.split('/')[-1].split('.')[0]
                    img_path_list.append((img_path, code_dir, code_name))
                    
        self.img_path_list = img_path_list
        self.transform = transform

        self.t5_feat_path = args.t5_feat_path
        self.short_t5_feat_path = args.short_t5_feat_path
        self.t5_feat_path_base = self.t5_feat_path.split('/')[-1]
        if self.short_t5_feat_path is not None:
            self.short_t5_feat_path_base = self.short_t5_feat_path.split('/')[-1]
        else:
            self.short_t5_feat_path_base = self.t5_feat_path_base
        self.image_size = args.image_size
        latent_size = args.image_size // args.downsample_size
        self.code_len = latent_size ** 2
        self.t5_feature_max_len = 120
        self.t5_feature_dim = 2048
        self.max_seq_length = self.t5_feature_max_len + self.code_len

    def __len__(self):
        return len(self.img_path_list)

    def dummy_data(self):
        img = torch.zeros((3, self.image_size, self.image_size), dtype=torch.float32)
        t5_feat_padding = torch.zeros((1, self.t5_feature_max_len, self.t5_feature_dim))
        attn_mask = torch.tril(torch.ones(self.max_seq_length, self.max_seq_length, dtype=torch.bool)).unsqueeze(0)
        valid = 0
        return img, t5_feat_padding, attn_mask, valid

    def __getitem__(self, index):
        # import ipdb; ipdb.set_trace()
        img_path, code_dir, code_name = self.img_path_list[index]
        try:
            img = Image.open(img_path).convert("RGB")                
        except:
            img, t5_feat_padding, attn_mask, valid = self.dummy_data()
            return img, t5_feat_padding, attn_mask, torch.tensor(valid)

        if min(img.size) < self.image_size:
            img, t5_feat_padding, attn_mask, valid = self.dummy_data()
            return img, t5_feat_padding, attn_mask, torch.tensor(valid)
        
        if self.transform is not None:
            img = self.transform(img) # torch.Size([3, 256, 256])
        
        t5_file = os.path.join(self.t5_feat_path, code_dir, f"{code_name}.npy")
        if torch.rand(1) < 0.3:
            t5_file = t5_file.replace(self.t5_feat_path_base, self.short_t5_feat_path_base)
        
        t5_feat_padding = torch.zeros((1, self.t5_feature_max_len, self.t5_feature_dim))
        assert os.path.isfile(t5_file)

        # import ipdb; ipdb.set_trace()
        if os.path.isfile(t5_file):
            try:
                t5_feat = torch.from_numpy(np.load(t5_file))
                t5_feat_len = t5_feat.shape[1] 
                # if t5_feat_len<self.t5_feature_max_len:
                #     import ipdb; ipdb.set_trace()
                feat_len = min(self.t5_feature_max_len, t5_feat_len)
                t5_feat_padding[:, -feat_len:] = t5_feat[:, :feat_len]
                emb_mask = torch.zeros((self.t5_feature_max_len,))
                emb_mask[-feat_len:] = 1
                attn_mask = torch.tril(torch.ones(self.max_seq_length, self.max_seq_length))
                T = self.t5_feature_max_len
                attn_mask[:, :T] = attn_mask[:, :T] * emb_mask.unsqueeze(0)
                eye_matrix = torch.eye(self.max_seq_length, self.max_seq_length)
                attn_mask = attn_mask * (1 - eye_matrix) + eye_matrix
                attn_mask = attn_mask.unsqueeze(0).to(torch.bool)
                valid = 1
            except:
                img, t5_feat_padding, attn_mask, valid = self.dummy_data()
        else:
            img, t5_feat_padding, attn_mask, valid = self.dummy_data()
            
        return img, t5_feat_padding, attn_mask, torch.tensor(valid)


class Text2ImgDatasetCode(Dataset):
    def __init__(self, args):
        pass




def build_t2i_image(args, transform):
    return Text2ImgDatasetImg(args.data_path, args.data_face_path, transform)

def build_t2i(args, transform):
    return Text2ImgDataset(args, transform)

def build_t2i_code(args):
    return Text2ImgDatasetCode(args)


from torch.utils.data import DataLoader
from torchvision import transforms
import argparse
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    # dataset & dataloader
    parser.add_argument("--dataset", type=str, required=True)
    # parser.add_argument("--data", type=str, required='')
    parser.add_argument("--video_meta_info_file", type=str, default='/storage/zhubin/liuyihang/add_aes/output/sucai_aes_1000.json')
    # parser.add_argument("--sample_rate", type=int, default=1)
    # parser.add_argument("--cache_dir", type=str, required='')
    parser.add_argument("--t5-model-path", type=str, default='./pretrained_models/t5-ckpt')
    parser.add_argument("--t5-model-type", type=str, default='flan-t5-xl')
    parser.add_argument("--max_height", type=int, default=1)
    parser.add_argument("--max_width", type=int, default=1)
    parser.add_argument("--precision", type=str, default='bf16')
    parser.add_argument("--model_max_length", type=int, default=512)
    parser.add_argument("--start_frame_ind", type=int, default=25)
    parser.add_argument("--num_frames", type=int, default=17)
    parser.add_argument("--data_root", type=str, default='/storage/dataset')
    parser.add_argument("--data-path", type=str, default='/storage/dataset')
    parser.add_argument("--t5-feat-path", type=str, required=True)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--short-t5-feat-path", type=str, default=None, help="short caption of t5_feat_path")
    parser.add_argument("--downsample-size", type=int, choices=[8, 16], default=16)
    args = parser.parse_args()

    
    transform = transforms.Compose([
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
    ])

    # tokenizer = AutoTokenizer.from_pretrained("/storage/ongoing/new/Open-Sora-Plan/cache_dir/mt5-xxl", cache_dir=args.cache_dir)
    assert os.path.exists(args.t5_model_path)
    device='cuda'
    precision = {'none': torch.float32, 'bf16': torch.bfloat16, 'fp16': torch.float16}[args.precision]

    dataset = Text2ImgDataset(args, transform)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0)
    for sample in dataloader:
        print(sample)
        # torch.Size([2, 3, 17, 480, 640]) torch.Size([2, 1, 512]) torch.Size([2, 1, 512])
        # print(sample['video_data']['video'].shape, sample['video_data']['input_ids'].shape, sample['video_data']['cond_mask'].shape)
    
    