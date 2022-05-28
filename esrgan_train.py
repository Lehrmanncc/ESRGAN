# -*- coding:utf-8 -*-
# Author : lehramnn
# Data : 5/27 16:00

import torch
import torchvision
import numpy as np
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from utils.util import psnr_score
from utils.util import init_weights
from utils.util import FeatureExtractor
from model.esrgan import RRDB_net, RaDiscriminator
from dataset.datasets import DIV2KDataset
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import visdom


def update_g(hr_data, lr_data, net_g, net_d, feature_net, optimizer_g, criterion, real_target, fake_target):
    percep_cri = criterion["percep_cri"]
    adversarial_cri = criterion["adversarial_cri"]
    content_cri = criterion["content_cri"]

    optimizer_g.zero_grad()
    fake_hr = net_g(lr_data)
    # 计算判别器对真假图像的分数
    score_fake = net_d(fake_hr)  # (batch_size, 1)
    score_real = net_d(hr_data)

    # 计算公式见原论文
    # 若Dra_rf -> 1代表真图像比假图像更真实
    Dra_rf = score_real - score_fake.mean()
    # 若Dra_fr -> 0代表假图像不如真实图像真实
    Dra_fr = score_fake - score_real.mean()

    content_loss = content_cri(fake_hr, hr_data)
    percep_loss = percep_cri(feature_net(fake_hr), feature_net(hr_data))

    # 生成器的目标是让：Dra_rf -> 0, Dra_fr -> 1
    # 即真图像不比假图像真实，假图像比真图像真实
    adversarial_loss_rf = adversarial_cri(Dra_rf, fake_target)
    adversarial_loss_fr = adversarial_cri(Dra_fr, real_target)
    adversarial_loss = (adversarial_loss_rf + adversarial_loss_fr) / 2

    loss_g = percep_loss + 0.005 * adversarial_loss + 0.01 * content_loss
    loss_g.backward()
    optimizer_g.step()
    return loss_g, fake_hr


def update_d(hr_data, fake_hr, net_d, optimizer_d, criterion, real_target, fake_target):
    percep_cri = criterion["percep_cri"]
    adversarial_cri = criterion["adversarial_cri"]
    content_cri = criterion["content_cri"]

    optimizer_d.zero_grad()
    score_real = net_d(hr_data)  # (batch_size, 1)
    score_fake = net_d(fake_hr.detach())

    Dra_rf = score_real - score_fake.mean()
    Dra_fr = score_fake - score_real.mean()

    # 判别器的目标是让：Dra_rf -> 1, Dra_fr -> 0
    # 即真图像比假图像真实，假图像不比真图像真实
    adversarial_loss_rf = adversarial_cri(Dra_rf, real_target)
    adversarial_loss_fr = adversarial_cri(Dra_fr, fake_target)
    loss_d = (adversarial_loss_rf + adversarial_loss_fr) / 2

    loss_d.backward()
    optimizer_d.step()
    return loss_d


def train(net_g, net_d, train_data, num_epoch, lr, device):
    net_g = net_g.to(device)
    net_d = net_d.to(device)
    feature_extractor = FeatureExtractor(torchvision.models.vgg19(pretrained=True), feature_layer=35).to(device)

    net_g.load_state_dict(torch.load("./weight/RRDBNet_final.pth"), strict=False)
    net_d.apply(init_weights)

    optimizer_hp = {"lr": lr, "betas": [0.9, 0.999]}
    optimizer_g = optim.Adam(net_g.parameters(), **optimizer_hp)
    optimizer_d = optim.Adam(net_d.parameters(), **optimizer_hp)

    scheduler_g = optim.lr_scheduler.MultiStepLR(optimizer_g,
                                                 milestones=[10, 20, 40, 60],
                                                 gamma=0.5)
    net_g.train()
    net_d.train()
    feature_extractor.eval()

    # 感知损失：计算生成的图像与高清图像分别经过VGG19网络后的特征图之间的损失
    # 内容损失：计算生成的图像与高清图像之间的损失
    criterion = {"percep_cri": nn.L1Loss().to(device),
                 "adversarial_cri": nn.BCEWithLogitsLoss().to(device),
                 "content_cri": nn.L1Loss().to(device)}
    viz = visdom.Visdom(env='RRDBnet')

    for epoch in range(1, num_epoch + 1):
        train_bar = tqdm(train_data)
        loss_g_sum = 0
        loss_d_sum = 0
        for i, (hr_data, lr_data) in enumerate(train_bar):
            hr_data, lr_data = hr_data.to(device), lr_data.to(device)
            # shape:[B, 1]
            real_target = torch.ones((hr_data.size(0), 1), requires_grad=False).to(device)
            fake_target = torch.zeros((hr_data.size(0), 1), requires_grad=False).to(device)

            loss_g, fake_hr = update_g(hr_data, lr_data, net_g, net_d, feature_extractor, optimizer_g, criterion,
                                       real_target, fake_target)
            loss_d = update_d(hr_data, fake_hr, net_d, optimizer_d, criterion, real_target, fake_target)

            train_bar.set_description(desc='Train [{}/{}] loss_g:{:.6f} loss_d:{:.6f}'.format(epoch,
                                                                                              num_epoch, loss_g.item(),
                                                                                              loss_d.item()))
            loss_g_sum += loss_g.item()
            loss_d_sum += loss_d.item()

        if epoch % 10 == 0:
            # 学习率变化
            scheduler_g.step()

        loss_g_mean = np.array([(loss_g_sum / len(train_data))])
        loss_d_mean = np.array([(loss_d_sum / len(train_data))])

        viz.line(X=np.array([epoch]), Y=loss_g_mean, win='loss win', name="loss_g", update='append', opts={
            'showlegend': True,  # 显示网格
            'title': "ESRGAN loss",
            'xlabel': "epoch",  # x轴标签
            'ylabel': "loss value",  # y轴标签
        })
        viz.line(X=np.array([epoch]), Y=loss_d_mean, win='loss win', name="loss_d", update='append')
    torch.save(net_g.state_dict(), "./weight/esgan_me.pth")


if __name__ == "__main__":
    net_g = RRDB_net(in_channel=3, feature_num=64, block_num=23, growth_num=32, gaussian_noise=True)
    net_d = RaDiscriminator(in_channel=3, feature_num=64)

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
    train_data = DataLoader(div2k_dataset, batch_size=4, shuffle=True, num_workers=2)

    num_epoch = 500
    lr = 1e-4

    use_cuda = torch.cuda.is_available()
    # 运行时的设备
    device = torch.device("cuda" if use_cuda else "cpu")
    train(net_g, net_d, train_data, num_epoch, lr, device)
