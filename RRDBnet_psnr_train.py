import torch
import torchvision
import numpy as np
from torch import nn
from torch.utils.data import DataLoader
from utils.util import psnr_score
from utils.util import init_weights_mini
from model.esrgan import RRDB_net
from dataset.datasets import DIV2KDataset
from torch.utils.data import DataLoader
from torchvision import transforms
from icecream import ic
import matplotlib.pyplot as plt
from tqdm import tqdm
import visdom


def train(net, train_data, num_epoch, lr, device):
    net = net.to(device)
    optimizer_hp = {"lr": lr, "betas": [0.9, 0.999]}
    optimizer = torch.optim.Adam(net.parameters(), **optimizer_hp)

    net.train()

    loss_fct = nn.L1Loss().to(device)
    viz = visdom.Visdom(env='RRDBnet')

    for epoch in range(1, num_epoch + 1):
        train_bar = tqdm(train_data)
        loss_sum = 0
        psnr_sum = 0
        for i, (hr_data, lr_data) in enumerate(train_bar):
            hr_data, lr_data = hr_data.to(device), lr_data.to(device)

            optimizer.zero_grad()
            fake_hr = net(lr_data)
            loss = loss_fct(fake_hr, hr_data)
            loss.backward()
            optimizer.step()

            psnr_batch_sum = 0
            for hr, fake in zip(hr_data, fake_hr):
                hr_img = hr.permute(1, 2, 0).cpu().detach().numpy()
                fake_img = fake.permute(1, 2, 0).cpu().detach().numpy()
                psnr_batch_sum += psnr_score(hr_img, fake_img)
            psnr_batch_mean = psnr_batch_sum / hr_data.shape[0]
            train_bar.set_description(desc='Train [{}/{}] loss:{:.6f} psnr:{:.2f}'.format(epoch, num_epoch, loss.item(),
                                                                                          psnr_batch_mean))
            loss_sum += loss
            psnr_sum += psnr_batch_mean
        loss_mean = np.array([(loss_sum / len(train_data)).cpu().detach().numpy()])
        psnr_mean = np.array([(psnr_sum / len(train_data))])

        viz.line(X=np.array([epoch]), Y=loss_mean, win='loss win', name="loss", update='append', opts={
            'showlegend': True,  # 显示网格
            'title': "RRDBnet loss",
            'xlabel': "epoch",  # x轴标签
            'ylabel': "loss value",  # y轴标签
        })
        viz.line(X=np.array([epoch]), Y=psnr_mean, win='psnr win', name="psnr", update='append')
    torch.save(rrdb_net.state_dict(), "./weight/RRDB_PSNR_me.pth")


if __name__ == "__main__":
    rrdb_net = RRDB_net()
    rrdb_net.apply(init_weights_mini)

    hr_patch_size = 128
    lr_patch_size = 32

    hr_path = "../data/DIV2K_HR/DIV2K_train_HR_sub"
    lr_path = "../data/DIV2K_HR/DIV2K_train_LR_bicubic/X4_sub"

    hr_transform = transforms.Compose([
        transforms.RandomCrop((hr_patch_size, hr_patch_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(degrees=90),
        transforms.ToTensor()
    ])
    lr_transform = transforms.Compose([
        transforms.RandomCrop((lr_patch_size, lr_patch_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(degrees=90),
        transforms.ToTensor()
    ])

    div2k_dataset = DIV2KDataset(hr_path, lr_path, hr_transform, lr_transform)
    train_data = DataLoader(div2k_dataset, batch_size=8, shuffle=True, num_workers=2)

    num_epoch = 100
    lr = 2e-4

    use_cuda = torch.cuda.is_available()
    # 运行时的设备
    device = torch.device("cuda" if use_cuda else "cpu")
    train(rrdb_net, train_data, num_epoch, lr, device)

