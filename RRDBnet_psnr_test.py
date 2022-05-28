import torch
import torchvision
import numpy as np
from .model.esrgan import RRDB_net
from torch import nn
from torch.utils.data import DataLoader

from torch.utils.data import DataLoader
from torchvision import transforms
from icecream import ic
import matplotlib.pyplot as plt
from tqdm import tqdm
import visdom
import os
import glob
import cv2


def test(net, test_path, result_path, device):
    net.eval()
    net = net.to(device)

    for i, path in enumerate(tqdm(glob.glob(test_path))):
        img_name = os.path.basename(path).split(".")[0]

        img = cv2.imread(path, cv2.IMREAD_COLOR)
        img = img * 1.0 / 255
        # 先将通道转为RGB，再将维度转为[c, h, w]
        img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
        img_lr = img.unsqueeze(0)
        img_lr = img_lr.to(device)

        with torch.no_grad():
            # .clamp_(0, 1)将输出的每个元素值压缩在0到1之间
            output = net(img_lr).data.squeeze().float().cpu().clamp_(0, 1).numpy()
        output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
        output = (output * 255.0).round()
        cv2.imwrite(os.path.join(result_path, '{:s}_rlt.png'.format(img_name)), output)


if __name__ == "__main__":
    test_path = "../data/DIV2K_HR/DIV2K_valid_LR_bicubic/X4/*"
    result_path = './results/RRDB_PSNR'
    model_path = "./weight/RRDBNet_final.pth"
    model = RRDB_net(3, 64, 23, 32)
    model.load_state_dict(torch.load(model_path), strict=False)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    # device = torch.device("cpu")
    test(model, test_path, result_path, device)


