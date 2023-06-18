# -*- coding: utf-8 -*-
"""
    @ProjectName: ConvBeamForming.py
    @File: modelTraining.py
    @Author: Chaos
    @Date: 2023/6/8
    @Description: 
"""
import time
import os
import pickle
import numpy as np
from scipy.io import savemat
import torch.cuda
import torch.nn as nn
from datasetGenerate import dataSplit


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()  # 创建基类 nn.Module 的 1 个实例
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


if __name__ == '__main__':
    print("================== modelTraining ==================")
    print("Start Time: " + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

    # ==================================== 导入数据集 ====================================
    filepath = ".\\dataset"
    filename = "M10SNR20.pkl"

    subFolderPath = os.path.join(filepath, filename.split(".")[0])

    # 第一次运行的时候分割数据集，往后直接调用
    if not os.path.isdir(subFolderPath):
        fileFullPath = os.path.join(filepath, filename)
        trainLoader, validation, _ = dataSplit(fileFullPath)
    else:
        files = os.listdir(subFolderPath)
        with open(os.path.join(subFolderPath, "trainLoader.pkl"), 'rb') as f:
            trainLoader = pickle.load(f)
        with open(os.path.join(subFolderPath, "validation.pkl"), 'rb') as f:
            validation = pickle.load(f)

    trainingDataLoader = torch.utils.data.DataLoader(trainLoader, batch_size=16, shuffle=True)

    # 初始化模型并转移到GPU上
    netModel = Net()
    if torch.cuda.is_available():
        netModel.cuda()

    # ==================================== 训练模型 ====================================
    # 定义损失函数（loss criterion）
    criterion = nn.CrossEntropyLoss()
    # 创建随机梯度下降优化器（optimizer）
    optimizer = torch.optim.SGD(netModel.parameters(), lr=0.001, momentum=0.9)

    print("Training Start: " + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    print("Model Training...")
    epochs = 50
    runningLossList = []
    lossList = []
    for epoch in range(epochs):
        runningLoss = 0.0
        for i, (inputs, bearingLabels) in enumerate(trainingDataLoader, 0):
            if torch.cuda.is_available():
                inputs, bearingLabels = inputs.cuda(), bearingLabels.cuda()
            optimizer.zero_grad()
            outputBearing = netModel(inputs)
            bearingLabels = torch.squeeze(bearingLabels, 1)  # 降维，可删除
            loss = criterion(outputBearing, bearingLabels)
            loss.backward()
            optimizer.step()
            runningLoss += loss.item()
        runningLossList.append(runningLoss / (i + 1))
        lossList.append(loss.data.tolist())
        print('Epoch %d, Running Loss: %.3f, Loss = %f.' % (epoch + 1, runningLoss / (i + 1), loss.data.tolist()))

    TrainingEndedTime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    print("Training Ended: " + TrainingEndedTime)

    # ==================================== 测试模型 ====================================
    # netModel = torch.load(".\\model\\netmodel.pth")

    correct = 0
    total = 0
    with torch.no_grad():
        for (inputs, bearingLabels) in validation:
            if torch.cuda.is_available():
                inputs, bearingLabels = inputs.cuda(), bearingLabels.cuda()
            outputBearing = netModel(inputs)
            _, predicted = torch.max(outputBearing.data, dim=1)
            total += bearingLabels.size(0)
            correct += (predicted == bearingLabels).sum().item()
    print('Accuracy of the network on the validation set: %d %%' % (100 * correct / total))

    # ==================================== 保存参数 ====================================
    savedName = ".\\model\\netmodel" + TrainingEndedTime + ".pth"
    torch.save(netModel, savedName)
    savedParam = ".\\model\\netmodelParam" + TrainingEndedTime + ".mat"
    savemat(savedParam, {"runningLossList":runningLoss, "lossList":lossList, "correctRate": (100 * correct / total)})

    print("================== プログラム終了 ==================")
