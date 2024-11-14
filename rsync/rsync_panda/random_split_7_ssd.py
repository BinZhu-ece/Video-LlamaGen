

# folders = '/storage/dataset/panda70m/panda70m_part*'
import glob, os, json
import numpy as np

# 获取所有匹配的文件夹
folders = glob.glob('/storage/dataset/panda70m/panda70m_part*')

# 将文件夹列表转换为numpy数组
folders_array = np.array(folders)

# 将数组平均分为7个部分
split_folders = np.array_split(folders_array, 7)

# 将每个部分转换为列表
split_folders_list = [list(part) for part in split_folders]

# 创建一个字典，键为列表的编号，值为列表内容
lists_dict = {f"List {i+1}": part for i, part in enumerate(split_folders_list)}

# 将字典保存到JSON文件
with open('folders_lists.json', 'w') as json_file:
    json.dump(lists_dict, json_file, indent=4)

print("Lists have been saved to folders_lists.json")