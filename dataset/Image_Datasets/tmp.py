import json
import jsonlines
import os
from tqdm import tqdm
# 读取 JSON 文件
with open('/storage/zhubin/LlamaGen/dataset/Image_Datasets/civitai_v1_1940032.json', 'r') as f:
    data = json.load(f)

# 创建 JSONL 文件
cnt = 0
os.makedirs('dataset/Image_Datasets/civitai_v1_10000', exist_ok=True)
with jsonlines.open('/storage/zhubin/LlamaGen/dataset/Image_Datasets/civitai_v1_10000.jsonl', mode='w') as writer:
    for item in tqdm(data):
        image_path = f"/storage/dataset/civitai/Images_civitai_v1/{item['path']}"
        target_path = f"/storage/zhubin/LlamaGen/dataset/Image_Datasets/civitai_v1_10000/{item['path']}"
        
        # os.system(f'cp {image_path} {target_path}')
        writer.write({'text':item['cap'][0], 'image_path':target_path})
        cnt += 1
        if cnt == 10000:
            break
