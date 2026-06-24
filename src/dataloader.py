import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
import ast
import random
from collections import Counter

class dataset(Dataset):
    def __init__(self, img_dir, csv_path, img_size=(224, 224), augment=False, balance=True):
        self.img_dir = img_dir
        self.df = pd.read_csv(csv_path)
        self.augment = augment
        self.balance = balance

        # 统计每个类别的样本数量（基于单标签用于平衡采样）
        self.label_counts = Counter(self.df['labels'])
        max_count = max(self.label_counts.values())

        # 过采样逻辑（基于单标签计数，但返回多标签向量）
        if balance:
            print(f"Balance enabled: {balance}")
            print("Original label distribution:", dict(self.label_counts))
            sampled_indices = []
            for label in self.label_counts:
                indices = self.df[self.df['labels'] == label].index.tolist()
                if len(indices) < max_count:
                    sampled_indices.extend(random.choices(indices, k=max_count))
                else:
                    sampled_indices.extend(indices)
            self.df = self.df.loc[sampled_indices].reset_index(drop=True)

            # 过采样后样本分布
            print("Label counts after balancing:", dict(Counter(self.df['labels'])))

        # 定义基础变换（调整大小 + 转换为 Tensor）
        base_transforms = [
            transforms.Resize(img_size),
            transforms.ToTensor(),
        ]

        # 数据增强模块
        if augment:
            self.transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),  # 随机水平翻转
                transforms.RandomRotation(30),  # 随机旋转 ±30 度
                transforms.ColorJitter(brightness=0.1),
                *base_transforms,
            ])
        else:
            self.transform = transforms.Compose(base_transforms)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # 加载图像
        img_name = os.path.join(self.img_dir, row['filename'])
        image = Image.open(img_name).convert('RGB')
        image = self.transform(image)

        # 解析多标签向量（target列为 "[1, 0, 0, 0, 0, 0, 0, 0]" 格式）
        target_str = row['target']
        multi_label = torch.tensor(ast.literal_eval(target_str), dtype=torch.float32)
        return image, multi_label








