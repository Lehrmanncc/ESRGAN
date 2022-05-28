import numpy as np
import cv2 as cv
import torch
from torch import nn


def psnr_score(hr_img, fake_img):
    if hr_img.shape != fake_img.shape:
        raise ValueError("Input image must have the same dimensions.")

    # 对生成的图片进行归一化
    fake_img_norm = np.zeros(fake_img.shape, dtype=fake_img.dtype)
    cv.normalize(fake_img, fake_img_norm, alpha=0, beta=1, norm_type=cv.NORM_MINMAX)

    mse = np.mean((hr_img - fake_img_norm) ** 2)
    psnr = 20 * np.log10(1.0 / np.sqrt(mse))
    return psnr


@torch.no_grad()
def init_weights_mini(m, scale=0.1):
    """
    权重初始化
    :param scale: Float, 初始权重放大或者缩小系数
    :param m: Module
    :return: Module, self
    """
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight)
        nn.init.constant_(m.bias, 0)
        m.weight.data *= scale
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight)
        nn.init.constant_(m.bias, 0)
        m.weight.data *= scale
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)


@torch.no_grad()
def init_weights(m, scale=1):
    """
    权重初始化
    :param scale (Float): Float, 初始权重放大或者缩小系数
    :param m (Module): Module
    :return: Module, self
    """
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight)
        nn.init.constant_(m.bias, 0)
        m.weight.data *= scale
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight)
        nn.init.constant_(m.bias, 0)
        m.weight.data *= scale
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)


class FeatureExtractor(nn.Module):
    """
    特征提取器，截取模型从前往后数特定的层
    :param model(torch.nn.Module): nn.Module
    :param feature_layer(int): default:12
    """

    def __init__(self, model, feature_layer=12):
        super(FeatureExtractor, self).__init__()
        self.features = nn.Sequential(*list(model.features.children())[:feature_layer]).eval()

    def forward(self, x):
        return self.features(x)
