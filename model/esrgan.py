import torch
from torch import nn
from torch.nn import functional as F


class GaussianNoise(nn.Module):
    def __init__(self, sigma=0.02, is_relative_detach=False):
        super().__init__()
        self.sigma = sigma
        self.is_relative_detach = is_relative_detach
        self.noise = torch.tensor(0, dtype=torch.float).to(torch.device('cuda'))

    def forward(self, x):
        if self.training and self.sigma != 0:
            scale = self.sigma * x.detach() if self.is_relative_detach else self.sigma * x
            sampled_noise = self.noise.repeat(*x.size()).normal_() * scale
            x = x + sampled_noise
        return x


class DenseBlock(nn.Module):
    def __init__(self, feature_num=64, growth_num=32, gaussian_noise=False):
        super(DenseBlock, self).__init__()
        self.gaussian_noise = gaussian_noise
        self.noise = GaussianNoise() if gaussian_noise else None

        self.conv1 = nn.Conv2d(feature_num, growth_num, 3, 1, 1)
        self.conv2 = nn.Conv2d(feature_num + growth_num * 1, growth_num, 3, 1, 1)
        self.conv3 = nn.Conv2d(feature_num + growth_num * 2, growth_num, 3, 1, 1)
        self.conv4 = nn.Conv2d(feature_num + growth_num * 3, growth_num, 3, 1, 1)
        self.conv5 = nn.Conv2d(feature_num + growth_num * 4, feature_num, 3, 1, 1)

        self.lrelu = nn.LeakyReLU(2e-2, inplace=True)
    def forward(self, x):
        y1 = self.lrelu(self.conv1(x))
        y2 = self.lrelu(self.conv2(torch.cat((y1, x), dim=1)))
        y3 = self.lrelu(self.conv3(torch.cat((y2, y1, x), dim=1)))
        y4 = self.lrelu(self.conv4(torch.cat((y3, y2, y1, x), dim=1)))
        y5 = self.conv5(torch.cat((y4, y3, y2, y1, x), dim=1))

        if self.gaussian_noise:
            return self.noise(y5 * 0.2 + x)
        else:
            return y5 * 0.2 + x


class RRDB(nn.Module):
    def __init__(self, feature_num=64, growth_num=32, gaussian_noise=False):
        super(RRDB, self).__init__()
        self.dense_block1 = DenseBlock(feature_num, growth_num, gaussian_noise)
        self.dense_block2 = DenseBlock(feature_num, growth_num, gaussian_noise)
        self.dense_block3 = DenseBlock(feature_num, growth_num, gaussian_noise)

    def forward(self, x):
        y = self.dense_block1(x)
        y = self.dense_block2(y)
        y = self.dense_block3(y)
        return x + 0.2 * y


def connect_block(block, block_num, **kwargs):
    net = []
    for _ in range(block_num):
        net.append(block(**kwargs))
    return nn.Sequential(*net)


class UpSample(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(UpSample, self).__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, 3, 1, 1)
        self.pix = nn.PixelShuffle(2)
        self.prelu = nn.PReLU()

    def forward(self, x):
        return self.prelu(self.pix(self.conv(x)))


class RRDB_net(nn.Module):
    def __init__(self, in_channel=3, feature_num=64, block_num=23, growth_num=32, gaussian_noise=False):
        super(RRDB_net, self).__init__()
        self.block_kwargs = {"feature_num": feature_num,
                             "growth_num": growth_num,
                             "gaussian_noise": gaussian_noise}
        self.conv1 = nn.Conv2d(in_channel, feature_num, 3, 1, 1)
        self.RRDBs = connect_block(RRDB, block_num, **self.block_kwargs)
        self.conv2 = nn.Conv2d(feature_num, feature_num, 3, 1, 1)
        self.up1 = UpSample(feature_num, 4*feature_num)
        self.up2 = UpSample(feature_num, 4*feature_num)
        self.conv3 = nn.Conv2d(feature_num, feature_num, 3, 1, 1)
        self.conv4 = nn.Conv2d(feature_num, in_channel, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=2e-1, inplace=True)

    def forward(self, x):
        out1 = self.conv1(x)
        out2 = self.RRDBs(out1)
        out3 = self.conv2(out2)
        out4 = self.up2(self.up1(out3))
        out = self.conv4(self.lrelu(self.conv3(out4)))
        return out


class RaDiscriminator(nn.Module):
    def __init__(self, in_channel, feature_num):
        super(RaDiscriminator, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, feature_num, 3, 1)

        self.conv2 = nn.Conv2d(feature_num, feature_num, 3, 2)
        self.bn1 = nn.BatchNorm2d(feature_num)

        self.conv3 = nn.Conv2d(feature_num, feature_num * 2, 3, 1)
        self.bn2 = nn.BatchNorm2d(feature_num * 2)

        self.conv4 = nn.Conv2d(feature_num * 2, feature_num * 2, 3, 2)
        self.bn3 = nn.BatchNorm2d(feature_num * 2)

        self.conv5 = nn.Conv2d(feature_num * 2, feature_num * 4, 3, 1)
        self.bn4 = nn.BatchNorm2d(feature_num * 4)

        self.conv6 = nn.Conv2d(feature_num * 4, feature_num * 4, 3, 2)
        self.bn5 = nn.BatchNorm2d(feature_num * 4)

        self.conv7 = nn.Conv2d(feature_num * 4, feature_num * 8, 3, 1)
        self.bn6 = nn.BatchNorm2d(feature_num * 8)

        self.conv8 = nn.Conv2d(feature_num * 8, feature_num * 8, 3, 2)
        self.bn7 = nn.BatchNorm2d(feature_num * 8)

        self.conv9 = nn.Conv2d(feature_num * 8, 1, 1, 1)

        self.lrelu = nn.LeakyReLU(negative_slope=2e-1, inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.lrelu(self.bn1(self.conv2(x)))
        x = self.lrelu(self.bn2(self.conv3(x)))
        x = self.lrelu(self.bn3(self.conv4(x)))
        x = self.lrelu(self.bn4(self.conv5(x)))
        x = self.lrelu(self.bn5(self.conv6(x)))
        x = self.lrelu(self.bn6(self.conv7(x)))
        x = self.lrelu(self.bn7(self.conv8(x)))
        x = self.conv9(x)
        x = F.avg_pool2d(x, x.shape[2:])
        return x.view(x.shape[0], -1)
