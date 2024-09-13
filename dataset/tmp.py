import argparse
import json
import os, io, csv, math, random
import numpy as np
import torchvision
from einops import rearrange
from decord import VideoReader
from os.path import join as opj
import gc
import torch
import torchvision.transforms as transforms
from torch.utils.data.dataset import Dataset
from tqdm import tqdm
from PIL import Image
import sys
current_file_path = os.path.abspath(__file__)
current_directory = os.path.dirname(current_file_path)
sys.path.append(os.path.join(current_directory, '..'))
from dataset.utils.dataset_utils import DecordInit
from dataset.utils.utils import text_preprocessing

def random_video_noise(t, c, h, w):
    vid = torch.rand(t, c, h, w) * 255.0
    vid = vid.to(torch.uint8)
    return vid

"""
video_meta_info_file
{
    "path": "mixkit/sunset/mixkit-lightbulb-in-snow-2569_resize1080p.mp4",
    "cap": [
      "The video depicts a serene and picturesque winter sunset scene throughout its duration. It begins with a tranquil atmosphere, featuring an illuminated light bulb resting on the snowy ground in the foreground, with a warm glow emanating from it. The background showcases a twilight sky blending hues of orange and blue, with silhouettes of bare trees visible against the horizon. Throughout the video, there are no noticeable changes or movements within the scene. The light bulb remains luminous, the sunset colors persist in their blend, and the tree silhouettes retain their position against the sky. This consistent imagery conveys a sense of stillness and tranquility, maintaining the winter evening's serene ambiance from start to finish."
    ],
    "size": 4302912,
    "duration": 15.5155,
    "resolution": {
      "width": 1920,
      "height": 1080
    },
    "frames": 372,
    "fps": 23.976023976023978,
    "aspect_ratio": "16:9",
    "motion": 0.9993534684181213,
    "motion_average": 0.9988254904747009
  },
"""

from torch.utils.data import Dataset, DataLoader

from language.t5 import T5Embedder

class T2V_dataset(Dataset):
    def __init__(self, args, transform, tokenizer):
 
        self.data_root = args.data_root
        self.num_frames = args.num_frames
        self.transform = transform
        self.tokenizer = tokenizer
        self.model_max_length = args.model_max_length
        self.v_decoder = DecordInit()
        # video_meta_info_file
        self.video_meta_info = self.read_jsonfile(args.video_meta_info_file)
        print(f'{args.video_meta_info_file=} is loaded successfully!')
        self.start_frame_ind = args.start_frame_ind # start from 1 s
        self.end_frame_ind = args.start_frame_ind + args.num_frames # 

        
    def read_jsonfile(self, jsonfile):
        with open(jsonfile, 'r', encoding='utf-8') as f:
            return json.load(f)
        
    def __len__(self):
        return len(self.video_meta_info)

    def __getitem__(self, idx):
        # import ipdb; ipdb.set_trace()
        try:
            video_data = self.get_video(idx)
            gc.collect()
            return dict(video_data=video_data)
        except Exception as e:
            return self.__getitem__(random.randint(0, self.__len__() - 1))
    def get_video(self, idx):
        video_path = os.path.join(self.data_root, self.video_meta_info[idx]['path'])
        # filter video seconds less than 2s, start_idx=25, end_idx=25+self.num_frames
        video = self.decord_read(video_path)
        video = self.transform(video)  # T C H W -> T C H W
        video = video.transpose(0, 1)  # T C H W -> C T H W
        text = random.choice(self.video_meta_info[idx]['cap'])

        text = text_preprocessing(text)
        text_tokens_and_mask = self.tokenizer(
            text,
            max_length=self.model_max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors='pt'
        )
        input_ids = text_tokens_and_mask['input_ids']
        cond_mask = text_tokens_and_mask['attention_mask']
        return dict(video=video, input_ids=input_ids, cond_mask=cond_mask)

    def get_image_from_video(self, video_data):
        select_image_idx = np.linspace(0, self.num_frames-1, self.use_image_num, dtype=int)
        assert self.num_frames >= self.use_image_num
        image = [video_data['video'][:, i:i+1] for i in select_image_idx]  # num_img [c, 1, h, w]
        input_ids = video_data['input_ids'].repeat(self.use_image_num, 1)  # self.use_image_num, l
        cond_mask = video_data['cond_mask'].repeat(self.use_image_num, 1)  # self.use_image_num, l
        return dict(image=image, input_ids=input_ids, cond_mask=cond_mask)
    def decord_read(self, path):
        decord_vr = self.v_decoder(path)
        # Sampling video frames
        frame_indice = np.linspace(self.start_frame_ind, self.end_frame_ind - 1, self.num_frames, dtype=int)
        video_data = decord_vr.get_batch(frame_indice).asnumpy()
        video_data = torch.from_numpy(video_data)
        video_data = video_data.permute(0, 3, 1, 2)  # (T, H, W, C) -> (T C H W)
        return video_data

from dataset.transform import ToTensorVideo, TemporalRandomCrop, RandomHorizontalFlipVideo, CenterCropResizeVideo, LongSideResizeVideo, SpatialStrideCropVideo
# from CausalVideoVAE.causalvideovae.model import ae_norm, ae_denorm
from torchvision.transforms import Lambda
ae_norm = {
    'CausalVAEModel_D4_2x8x8': Lambda(lambda x: 2. * x - 1.),
    'CausalVAEModel_D8_2x8x8': Lambda(lambda x: 2. * x - 1.),
    'CausalVAEModel_D4_4x8x8': Lambda(lambda x: 2. * x - 1.),
    'CausalVAEModel_D8_4x8x8': Lambda(lambda x: 2. * x - 1.),
}
from transformers import AutoTokenizer
def tmp():
    import json
    # 读取json文件
    with open('/storage/zhubin/liuyihang/add_aes/output/sucai_aes.json', 'r') as file:
        data = json.load(file)
    # 提取前10个元素
    new_data = data[:1000]
    # 将新的列表写入新的json文件
    with open('/storage/zhubin/liuyihang/add_aes/output/sucai_aes_1000.json', 'w') as file:
        json.dump(new_data, file, ensure_ascii=False, indent=4)



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    # dataset & dataloader
    # parser.add_argument("--dataset", type=str, required=True)
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
    
    args = parser.parse_args()

    resize = [CenterCropResizeVideo((args.max_height, args.max_width)), ]
    # norm_fun = ae_norm[args.ae]
    norm_fun = Lambda(lambda x: 2. * x - 1.)
    transform = transforms.Compose([
        ToTensorVideo(),
        *resize, 
        # RandomHorizontalFlipVideo(p=0.5),  # in case their caption have position decription
        norm_fun
    ])
    # tokenizer = AutoTokenizer.from_pretrained("/storage/ongoing/new/Open-Sora-Plan/cache_dir/mt5-xxl", cache_dir=args.cache_dir)
    assert os.path.exists(args.t5_model_path)
    device='cuda'
    precision = {'none': torch.float32, 'bf16': torch.bfloat16, 'fp16': torch.float16}[args.precision]

    t5_xxl = T5Embedder(
        device=device, 
        local_cache=True, 
        cache_dir=args.t5_model_path, 
        dir_or_name=args.t5_model_type,
        torch_dtype=precision
    )
    dataset = T2V_dataset(args, transform, t5_xxl.tokenizer)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0)
    for sample in dataloader:
        print(sample['video_data'])
    
