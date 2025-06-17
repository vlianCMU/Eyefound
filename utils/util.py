import os
import cv2
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, Dataset
from monai import transforms

class FundusDataset(Dataset):
    def __init__(self, data_root, metadata_path, transform=None, section="train") -> None:
        """
        初始化眼底图像数据集
        
        参数:
            data_root (str): 数据根目录（包含train、test、validation子目录）
            metadata_path (str): metadata.csv文件路径
            transform (callable): 图像变换函数
            section (str): 数据集分区（"train", "test", "validation"）
        """
        self.data_root = data_root
        self.section = section
        self.transform = transform
        
        # 读取metadata.csv文件
        self.metadata = pd.read_csv(metadata_path)
        
        # 构建图像路径和描述的映射
        self.image_paths = []
        self.descriptions = []
        
        # 图像目录路径
        image_dir = os.path.join(data_root, section)
        
        # 遍历metadata中的每一行
        for idx, row in self.metadata.iterrows():
            img_path = os.path.join(image_dir, row['file_name'])
            if os.path.exists(img_path):
                self.image_paths.append(img_path)
                self.descriptions.append(row['text'])
        
        print(f"Loaded {len(self.image_paths)} images from {section} set")

    def __len__(self):
        """返回数据集大小"""
        return len(self.image_paths)
    
    def read_image(self, fp):
        """
        读取JPG图像并进行预处理
        
        参数:
            fp (str): 图像文件路径
            
        返回:
            np.ndarray: 处理后的图像数组，形状为(C,H,W)
        """
        # 读取图像
        img = cv2.imread(fp)
        
        # 检查图像是否成功读取
        if img is None:
            raise ValueError(f"无法读取图像: {fp}")
            
        # 转换颜色空间从BGR到RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # 确保图像是uint8格式
        if img.dtype != np.uint8:
            img = img.astype(np.uint8)
        
        # 转换为float32但保持在[0,255]范围，让transforms处理归一化
        img = img.astype(np.float32)
        
        # 转换为PyTorch格式 (C,H,W)
        img = np.transpose(img, (2,0,1))
    
        return img

    def _transform(self, fp):
        """
        应用变换到图像
        
        参数:
            fp (str): 图像文件路径
            
        返回:
            torch.Tensor: 变换后的图像张量
        """
        data_i = self.read_image(fp)
        return self.transform(data_i)

    def __getitem__(self, index):
        """
        获取数据集中的单个项目
        
        参数:
            index (int): 索引
            
        返回:
            tuple: (图像张量, 文本描述)
        """
        img_path = self.image_paths[index]
        text = self.descriptions[index]
        
        # 随机概率决定是否使用简化描述
        # if np.random.random() <= 0.05:
        #     text = 'a colored fundus image'  # 简化描述，类似于原代码中的'freedom'
            
        return self._transform(img_path), text

def get_dl(config):
    """
    获取数据加载器
    
    参数:
        config (object): 配置对象，包含训练参数
        
    返回:
        tuple: (数据加载器, 数据集)
    """
    # 定义训练数据变换 - 修复数据范围问题
    train_transforms = transforms.Compose([
            # transforms.RandScaleCrop([0.9,0.9],[1.1,1.1],random_size=True),
            transforms.Resize([config.image_size,config.image_size]),
            transforms.RandFlip(prob=0.5, spatial_axis=0),
            transforms.RandFlip(prob=0.5, spatial_axis=1),
            # transforms.RandRotate90(prob=0.3),
            
            # 关键修复：确保正确的数据范围转换
            transforms.ScaleIntensity(minv=0.0, maxv=1.0),  # 先归一化到[0,1]
            transforms.ToTensor(),
            # transforms.RandAdjustContrast(prob=0.1, gamma=(0.97, 1.03)),
            transforms.NormalizeIntensity(0.5, 0.5),  # 然后转换到[-1,1]
        ])

    # 创建数据集和数据加载器
    data_root = config.data_root  # 从配置中获取数据根目录
    metadata_path = os.path.join(data_root, "train", "metadata.csv")  # 训练集的metadata路径
    
    dataset = FundusDataset(
        data_root=data_root,
        metadata_path=metadata_path,
        transform=train_transforms,
        section="train"
    )
    
    train_dataloader = DataLoader(
        dataset, 
        batch_size=config.train_batch_size, 
        shuffle=True, 
        drop_last=True, 
        num_workers=config.num_workers,
        pin_memory=True,
        prefetch_factor=2
    )

    return train_dataloader, dataset