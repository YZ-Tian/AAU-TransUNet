import numpy as np
from torch.utils.data import Dataset
import os
from PIL import Image
from torchvision import transforms
import torch


class CarvanaDataset(Dataset):
    def __init__(self, images_dir, labels_dir, transform=None):
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.transform = transform
        self.images = os.listdir(self.images_dir)

        # 图像归一化转换
        self.image_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),  # 自动归一化到 [0,1]
            transforms.Normalize(mean=[0.5], std=[0.5])  # 归一化到 [-1,1]
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        images_path = os.path.join(self.images_dir, self.images[index])
        labels_path = os.path.join(self.labels_dir, self.images[index])

        # 读取图像和标签（都作为灰度图）
        image = Image.open(images_path).convert("L")
        label = Image.open(labels_path).convert("L")

        # resize 图像和标签
        image = image.resize((256, 256), Image.BILINEAR)
        label = label.resize((256, 256), Image.NEAREST)  # 标签必须用最近邻，不然会产生中间值

        # 转换为numpy数组
        image = np.array(image)
        label = np.array(label)

        # 将灰度标签映射为类别（根据你的实际需求调整）
        # 示例：假设灰度值0=背景，128=类别1，255=类别2
        label = np.where(label == 128, 1, label)  # 128 → 1
        label = np.where(label == 255, 2, label)  # 255 → 2

        # 转换为Tensor
        image = transforms.ToTensor()(image)  # [1, H, W]
        label = torch.from_numpy(label).long()  # [H, W]

        if self.transform is not None:
            augmentation = self.transform(image=image, label=label)
            image = augmentation["image"]
            label = augmentation["label"]

        return image, label

# labels_path = r"D:\Desktop\Task03_Liver_png_contrast\labels_standard\labels_standard_validation\liver_000.nii_slice060.png"
# print(np.unique(np.array(Image.open(labels_path).convert("L"))))