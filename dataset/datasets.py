import os
from torch.utils.data import Dataset
from PIL import Image


class DIV2KDataset(Dataset):
    def __init__(self, hr_path, lr_path, hr_transforms, lr_transforms):
        """
        DIV2K数据集的dataset, 需先运行extract_subimages.py文件，将图像进行分割
        :param hr_path: 高清图像分割后的路径
        :param lr_path: 低清图像分割后的路径，这里用的低清图像为X4
        :param hr_transforms: 高清图像的transform
        :param lr_transforms: 低清图像的transform
        """
        self.hr_path = hr_path
        self.lr_path = lr_path

        self.hr_file_list = os.listdir(hr_path)
        self.lr_file_list = os.listdir(lr_path)

        self.hr_transforms = hr_transforms
        self.lr_transforms = lr_transforms

    def __getitem__(self, item):
        hr_file = self.hr_file_list[item]
        lr_file = self.lr_file_list[item]

        hr_img = Image.open(os.path.join(self.hr_path, hr_file))
        lr_img = Image.open(os.path.join(self.lr_path, lr_file))

        hr_img = self.hr_transforms(hr_img)
        lr_img = self.lr_transforms(lr_img)
        # hr_img.shape:[128, 128]
        # lr_img.shape:[32, 32]

        return hr_img, lr_img

    def __len__(self):
        return len(self.hr_file_list)
