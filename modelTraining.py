# -*- coding: utf-8 -*-
"""
    @ProjectName: ConvBeamForming.py
    @File: modelTraining.py
    @Author: Chaos
    @Date: 2023/6/8
    @Description: 
"""
import os, pickle
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()  # 创建基类 nn.Module 的 1 个实例
        # 根据架构图创建全连接层
        self.fc1 = nn.Linear(28 * 28, 200)
        self.fc2 = nn.Linear(200, 200)
        self.fc3 = nn.Linear(200, 10)


if __name__ == '__main__':
    print("================== modelTraining ==================")

    # 导入数据
    datasetPath = ".\\dataset"
    datasetFilename = "M10SNR20.pkl"
    dataset = os.path.join(datasetPath, datasetFilename)
    with open(dataset, 'rb') as f:
        data = pickle.load(f)

    # 搭建模型

    print("================== プログラム終了 ==================")
