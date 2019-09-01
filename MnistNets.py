import torch.nn as nn
import torch
import numpy as np


class EncoderNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(1, 128, 3, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),  # N,128,14,14
            nn.Conv2d(128, 256, 3, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),  # N,256,7,7
            nn.Conv2d(256, 512, 3, 2, 0),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),  # N,512,3,3
            nn.Conv2d(512, 256, 3, 1, 0),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),  # N,256,1,1
            nn.Conv2d(256, 128, 1, 1, 0),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),  # N,128,1,1
            nn.Conv2d(128, 2, 1, 1, 0),  # N,2,1,1
        )

    def forward(self, x):
        out = self.layer(x)
        logsigma = out[:, :1, :, :]
        miu = out[:, 1:, :, :]
        return logsigma, miu


class DecoderNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Sequential(
            nn.ConvTranspose2d(128, 128, 1, 1, 0, ),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),  # N,128,1,1
            nn.ConvTranspose2d(128, 256, 1, 1, 0, ),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),  # N,256,1,1
            nn.ConvTranspose2d(256, 512, 3, 2, 0),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),  # N,512,3,3
            nn.ConvTranspose2d(512, 256, 3, 2, 0),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),  # N,256,7,7
            nn.ConvTranspose2d(256, 128, 3, 2, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),  # N,128,14,14
            nn.ConvTranspose2d(128, 1, 3, 2, 1, 1),  # N,2,1,1
            nn.Sigmoid()
        )

    def forward(self, x, logsigma, miu):
        # print(x.size())
        # print(logsigma.size())
        # print(miu.size())
        x = x * torch.exp(logsigma) + miu
        x = x.permute(0, 3, 1, 2)  # 自己制作的数据是N,H,W,C,需要先进行转置操作
        # print(x.size())
        out = self.layer(x)
        return out


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = EncoderNet()
        self.decoder = DecoderNet()

    def getloss(self, logsigma, miu):
        kl_loss = torch.mean(- torch.log(logsigma ** 2) + (miu ** 2) + (logsigma ** 2) - 1) * 0.5
        return kl_loss

    def forward(self, x, c):
        logsigma, miu = self.encoder(x)
        # print('logsigma', torch.mean(torch.exp(logsigma)))
        # print('miu', torch.mean(torch.exp(miu)))
        kl_loss = self.getloss(logsigma, miu)
        output = self.decoder(c, logsigma, miu)
        return output, kl_loss
