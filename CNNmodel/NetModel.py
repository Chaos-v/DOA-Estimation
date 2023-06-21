# -*- coding: utf-8 -*-
"""
    @File: NetModel.py
    @Author: Chaos
    @Date: 2023/6/8
    @Description: 网络模型
"""
import torch.nn as nn


class NetM10(nn.Module):
    """
    阵元数量 M = 10，阵元间距等于半波长的网络模型。
    网络结构复现自文献：《DOA estimation based on CNN for underwater acoustic array》
    """
    def __init__(self):
        super(NetM10, self).__init__()  # 创建基类 nn.Module 的 1 个实例
        # 根据架构图创建全连接层
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=32, kernel_size=2, stride=1, padding='same'),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=2, stride=1, padding='same'),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=1)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding='same'),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=3)
        )
        self.fc = nn.Sequential(
            nn.Linear(in_features=576, out_features=1024),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=1024, out_features=512),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=512, out_features=181)
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = x.view(-1, 576)  # 返回新张量，数据相同，形状不同
        x = self.fc(x)
        return x


class NetM20(nn.Module):
    """
    阵元数量 M = 20，阵元间距等于半波长的网络模型。
    网络结构复现自文献：《DOA estimation based on CNN for underwater acoustic array》
    """
    def __init__(self):
        super(NetM20, self).__init__()  # 创建基类 nn.Module 的 1 个实例
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=32, kernel_size=2, stride=1, padding='same'),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=1)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=2, stride=1, padding='same'),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=1, stride=1)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=2, stride=1, padding='same'),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=2, stride=1, padding='same'),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.fc = nn.Sequential(
            nn.Linear(in_features=10368, out_features=1024),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=1024, out_features=512),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=512, out_features=181)
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = x.view(-1, 10368)  # 返回新张量，数据相同，形状不同
        x = self.fc(x)
        return x


if __name__ == '__main__':
    print("================== modelTraining ==================")

    print("================== プログラム終了 ==================")
