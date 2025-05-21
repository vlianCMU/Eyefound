import os
import pickle
import json
import numpy as np
from PIL import Image
import pandas as pd

# 创建目录
os.makedirs('fundus_pretrain', exist_ok=True)

# 假设眼底图像位于这个路径
fundus_data_path = '/data/home/yantao/diffusers/examples/text_to_image/cataract_classifier_dataset'

# 处理训练集图像，转换为LCTFound可用的格式
all_image_paths = []
iid_map_info = {}

for folder in ['train', 'test', 'validation']:
    folder_path = os.path.join(fundus_data_path, folder)
    metadata_path = os.path.join(folder_path, 'metadata.csv')
    
    if os.path.exists(metadata_path):
        df = pd.read_csv(metadata_path)
        
        # 为每张图片创建.npy文件
        for idx, row in df.iterrows():
            img_path = os.path.join(folder_path, row['file_name'])
            
            if os.path.exists(img_path):
                # 处理图像名作为唯一ID
                image_id = os.path.splitext(row['file_name'])[0]
                
                # 保存疾病描述到iid_map_info
                iid_map_info[image_id] = row['text']
                
                # 读取图像
                img = Image.open(img_path)
                img_array = np.array(img)
                
                # 确保图像是RGB格式
                if len(img_array.shape) == 2:
                    # 如果是灰度图，转为RGB
                    img_array = np.stack([img_array, img_array, img_array], axis=2)
                
                # 创建目标目录
                os.makedirs(f'fundus_pretrain/{image_id}', exist_ok=True)
                
                # 保存为.npy格式，使用LCTFound期望的格式
                npy_path = f'fundus_pretrain/{image_id}/{image_id}.npy'
                np.savez_compressed(npy_path, im=img_array)
                
                # 保存路径到列表
                all_image_paths.append(npy_path)

# 保存所有图像路径到pickle文件
with open('total_file_path.pkl', 'wb') as f:
    pickle.dump(all_image_paths, f)

# 保存iid_map_info到json文件
with open('pid_map_info.json', 'w', encoding='utf-8') as f:
    json.dump(iid_map_info, f, ensure_ascii=False)

print(f"Total images processed: {len(all_image_paths)}")