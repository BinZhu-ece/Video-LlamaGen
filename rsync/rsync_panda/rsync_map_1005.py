
import os
from multiprocessing import Pool



# 定义需要同步的文件夹
def sync_folder(dir_path):
    # 目标目录
    target_dir = '/data1/panda70m'
    os.makedirs(target_dir, exist_ok=True)
    source_folder = dir_path
    os.system(f'rsync -avz --progress --include="*/" --include="*.mp4"   --exclude="*" {source_folder} {target_dir}')


def sync_folder2(dir_path):
    target_dir = '/data2/panda70m'
    os.makedirs(target_dir, exist_ok=True)
    source_folder = dir_path
    os.system(f'rsync -avz --progress --include="*/" --include="*.mp4"   --exclude="*" {source_folder} {target_dir}')

def sync_folder3(dir_path):
    target_dir = '/data3/panda70m'
    os.makedirs(target_dir, exist_ok=True)
    source_folder = dir_path
    os.system(f'rsync -avz --progress --include="*/" --include="*.mp4"   --exclude="*" {source_folder} {target_dir}')



def read_json(jsonfile):
    with open(jsonfile, 'r') as f:
        data = json.load(f)
    return data

import json
def cal_size_total(list1):
    folder_path = "/storage/dataset/panda70m"  
    # 获取所有部分文件夹的路径  
    part_paths = [os.path.join(folder_path, f'{part_index:05d}') for part_index in list1]  
 
    folder_sizes = 0

    # import ipdb; ipdb.set_trace()
    for part_path in part_paths:
        size_file = part_path+'_size.json'
        if os.path.exists(size_file):
            folder_size = read_json(size_file)
            if 'M' in folder_size:
                folder_sizes += eval(folder_size[:-1])/1024 + 0.1
            elif 'G' in folder_size:
                folder_sizes += eval(folder_size[:-1])+ 0.1
            elif 'T' in folder_size:
                folder_sizes += eval(folder_size[:-1])*1024
            else:
                print(part_path, folder_size, 'error!')
                folder_sizes += 0
                # raise KeyError
    print(f"Total size of folders: {folder_sizes} GB, folder nums:{len(part_paths)}")        
    # assert folder_sizes < 6800
    
# 多进程处理

 

if __name__ == "__main__":
    import argparse

    # 创建一个 ArgumentParser 对象
    parser = argparse.ArgumentParser(description='A simple example of argparse usage')
    # 添加一个位置参数
    parser.add_argument('--list_index', type=int, help='Input file to process')
    parser.add_argument('--node', type=str, help='Input file to process')
    parser.add_argument('--disk', type=str, help='Input file to process')
    # 添加一个可选参数
    # 解析命令行参数
    args = parser.parse_args()


    # import ipdb; ipdb.set_trace()
    with open('/storage/zhubin/LlamaGen/rsync/rsync_panda/folders_lists.json', 'r', encoding='utf-8') as f:
        folder_list = json.load(f)[f"List {args.list_index}"]


    #  077 data1
    if args.node=='077' and args.disk=='data1':
        with Pool(processes=128) as pool:
            pool.map(sync_folder, folder_list)
    """ 
    
    python3  /storage/zhubin/LlamaGen/rsync/rsync_panda/rsync_map_1005.py --node 077 --disk data1 --list_index 1 
    
    """

    #  077 data2 
    if args.node=='077' and args.disk=='data2':
        with Pool(processes=128) as pool:
            pool.map(sync_folder2, folder_list)
    """ 
    
    python3  /storage/zhubin/LlamaGen/rsync/rsync_panda/rsync_map_1005.py --node 077 --disk data2 --list_index 2
    
    """

    #  077 data3
    if args.node=='077' and args.disk=='data3':
        with Pool(processes=128) as pool:
            pool.map(sync_folder3, folder_list)
    """ 
    python3  /storage/zhubin/LlamaGen/rsync/rsync_panda/rsync_map_1005.py --node 077 --disk data3 --list_index 3
    """
    

    #  103 data1
    if args.node=='103' and args.disk=='data1':
        with Pool(processes=128) as pool:
            pool.map(sync_folder, folder_list)
    """ 
    python3  /storage/zhubin/LlamaGen/rsync/rsync_panda/rsync_map_1005.py --node 103 --disk data1  --list_index 4
    """

    #  103 data2
    if args.node=='103' and args.disk=='data2':
        with Pool(processes=128) as pool:
            pool.map(sync_folder2, folder_list)
    """ 
    python3  /storage/zhubin/LlamaGen/rsync/rsync_panda/rsync_map_1005.py --node 103 --disk data2  --list_index 5
    """