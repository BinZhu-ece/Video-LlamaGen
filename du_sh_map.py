import os
import subprocess
import json
from multiprocessing import Pool, cpu_count

# 定义要遍历的文件夹路径
base_dir = '/storage/dataset/recap_datacomp_1b_data/output'
start_index = 0
end_index = 2719

# 定义获取文件夹大小的函数
def get_folder_size(folder_index):
    folder_name = f'{folder_index:05d}'  # 格式化为五位数
    folder_path = os.path.join(base_dir, folder_name)
    
    if os.path.exists(folder_path):

        save_file = folder_path+'_size.json'

        if os.path.exists(save_file):
            print(f"File {save_file} already exists.")
            return folder_path, None
        # 使用 subprocess 执行 du -sh 命令获取文件夹大小
        result = subprocess.run(['du', '-sh', folder_path], stdout=subprocess.PIPE)
        size = result.stdout.decode('utf-8').split()[0]  # 提取文件大小
        print(folder_path, size)

        
        with open(save_file, 'w', encoding='utf-8') as f:
            json.dump(str(size), f, indent=2)
        return folder_path, size
    else:
        print(f"Folder {folder_path} does not exist.")
        return folder_path, None


# 多进程处理
if __name__ == "__main__":

    # 使用进程池并发执行
    with Pool(processes=cpu_count()) as pool:
        results = pool.map(get_folder_size, range(start_index, end_index + 1))

    # 过滤掉 None 的结果，并构建字典
    folder_sizes = {folder_path: size for folder_path, size in results if size is not None}

    # 将结果保存为 JSON 文件
    output_file = 'folder_sizes.json'
    with open(output_file, 'w') as f:
        json.dump(folder_sizes, f, indent=4)

    print(f"Folder sizes saved to {output_file}")


    """
    
    cd /storage/zhubin/LlamaGen/
    python du_sh_map.py
    
    """
