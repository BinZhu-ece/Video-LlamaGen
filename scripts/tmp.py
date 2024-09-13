import json
import shutil

# 读取JSON文件
with open('/storage/zhubin/liuyihang/add_aes/output/sucai_aes.json', 'r') as f:
    data = json.load(f)

# 创建一个新的JSON文件用于保存captions
with open('video_captions.json', 'w') as f:
    json.dump({}, f)

# 遍历JSON文件中的每个item
import os
new_data = []

cnt = 0
for item in data:
    # 使用 shutil.copy2 将视频文件从源路径复制到当前路径

    if 'istock' in item['path']:

        new_item = {}
        shutil.copy2(item['path'], '/storage/zhubin/LlamaGen/sucai_subset/')

        # 更新新的JSON文件
        new_item['name'] = os.path.basename(item['path'])
        new_item['caption'] = item['cap']

        new_data.append(new_item)
        cnt += 1
        if cnt == 20:
            break

# 保存新的JSON文件
with open('/storage/zhubin/LlamaGen/sucai_subset/info_data', 'w') as f:
    json.dump(new_data, f)
