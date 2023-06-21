# -*- coding: utf-8 -*-
"""
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
import matplotlib.pyplot as plt
import torch.cuda
import torch.nn as nn
# 导入网络模型对象
from CNNmodel.NetModel import NetM20


def dataSplit(fpath):
    """加载同时预处理数据集，只针对特定数据生效，并且做一个分割，分割成训练集，验证集，测试集并保存"""
    with open(fpath, 'rb') as f:
        data = pickle.load(f)

    # ==================================== 数据集预处理 ====================================
    dataset = [None] * len(data)
    tmpData = np.zeros((2, 20, 20), dtype=float)
    thetaList = list(range(-90, 91, 1))
    for i in range(len(data)):
        tmpData[0] = data[i][0].real
        tmpData[1] = data[i][0].imag
        index = thetaList.index(data[i][1])
        tmpBearing = np.zeros((1, 181), dtype=int)
        tmpBearing[:, index] = 1
        dataset[i] = (torch.Tensor(tmpData), torch.Tensor(tmpBearing))

    training = [None] * 101903
    validate = 33847 * [None]
    test = 45250 * [None]
    for i in range(181):
        index = i * 1000
        training[563 * i:563 * (i + 1)] = dataset[index:index + 563]
        validate[187*i:187*(i+1)] = dataset[index+563:index + 750]
        test[250*i:250*(i+1)] = dataset[index+750:index+1000]

    # 保存数据集
    (dataPath, dataName) = os.path.split(fpath)
    subFolderName = dataName.split(".")[0]
    folderPath = os.path.join(dataPath, subFolderName)
    if not os.path.isdir(folderPath):
        os.makedirs(folderPath)

    with open(os.path.join(folderPath, "trainLoader.pkl"), 'wb') as f:
        pickle.dump(training, f)
    with open(os.path.join(folderPath, "validation.pkl"), 'wb') as f:
        pickle.dump(validate, f)
    with open(os.path.join(folderPath, "testLoader.pkl"), 'wb') as f:
        pickle.dump(test, f)

    return training, validate, test


if __name__ == '__main__':
    print("================== modelTraining ==================")
    print("Start Time: " + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

    # ==================================== 导入数据集 ====================================
    filepath = ".\\dataset"
    filename = "M20SNR20.pkl"

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
    netModel = NetM20()
    if torch.cuda.is_available():
        netModel.cuda()

    # ==================================== 训练模型 ====================================
    # 定义损失函数（loss criterion）
    criterion = nn.CrossEntropyLoss()
    # 创建随机梯度下降优化器（optimizer）
    optimizer = torch.optim.SGD(netModel.parameters(), lr=0.001, momentum=0.8)

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

            bearingLabels = torch.squeeze(bearingLabels, 1)  # 降维，可删除，仅当前模型使用

            optimizer.zero_grad()  # 初始化参数grad的值
            outputBearing = netModel(inputs)
            loss = criterion(outputBearing, bearingLabels)  # 使用交叉熵计算损失
            loss.backward()  # 反向传播
            optimizer.step()  # 更新随机梯度下降参数
            runningLoss += loss.item()  # runningLoss用于统计loss平均值
        # 设置列表存放损失的平均值和损失值
        runningLossList.append((runningLoss / (i + 1)))
        lossList.append(loss.data.tolist())
        print('Epoch %d, Running Loss: %.3f, Loss = %f.' % (epoch + 1, runningLoss / (i + 1), loss.data.tolist()))

    TrainingEndedTime = time.strftime("%Y%m%d%H%M", time.localtime())
    print("Training Ended: " + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

    # ==================================== 验证模型 ====================================
    correct = 0
    total = 0
    with torch.no_grad():
        for (inputs, bearingLabels) in validation:
            if torch.cuda.is_available():
                inputs, bearingLabels = inputs.cuda(), bearingLabels.cuda()
            outputBearing = netModel(inputs)
            _, predicted = torch.max(outputBearing.data, dim=1)  # 输出预测的向量中最大值的位置
            _, reality = torch.max(bearingLabels, dim=1)
            if predicted == reality:
                correct += 1
            total += 1
    #         total += bearingLabels.size(0)
    #         correct += (predicted == bearingLabels).sum()
    # accuracyRate = (100 * torch.true_divide(correct, total)).item()
    accuracyRate = 100 * correct / total  # 计算正确率
    print('Accuracy of the network on the validation set: %d %%' % (accuracyRate))

    # ==================================== 保存参数 ====================================
    if accuracyRate >= 50:
        savedName = ".\\model\\netmodel" + TrainingEndedTime + ".pth"
        torch.save(netModel, savedName)
        savedParam = ".\\model\\netmodelParam" + TrainingEndedTime + ".mat"
        savemat(savedParam,
                {"runningLossList": runningLossList, "lossList": lossList, "correctRate": (100 * correct / total)})
        # 损失图
        plt.switch_backend('agg')
        plt.plot(list(range(len(runningLossList))), runningLossList)
        plt.plot(list(range(len(lossList))), lossList)
        savePicName = ".\\model\\netmodelPic" + TrainingEndedTime + ".jpg"
        plt.savefig(savePicName)
    else:
        print("Low accuracy, not saved")

    print("================== プログラム終了 ==================")
