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
from pathLoader import getpath
from datasetSplit import dataSplit
# 导入网络模型对象
from CNNmodel.NetModel import NetM32
# from CNNmodel.NetModel import NetM20
# from CNNmodel.NetModel import NetM10
from tqdm import tqdm


if __name__ == '__main__':
    print('=' * 51 , "\n================== modelTraining ==================")
    print("Program Start Time at: " + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

    # ==================================== 导入数据集 ====================================
    datasetPath = getpath("datasetPath").split("\\")
    filepath = ''
    for i in range(len(datasetPath)):
        if ".pkl" in datasetPath[i]:
            for tmp in datasetPath[:i]:
                filepath = os.path.join(filepath,tmp)
            filename = datasetPath[i]
    
    print(filename)

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

    batchSize = 16
    trainingDataLoader = torch.utils.data.DataLoader(trainLoader, batch_size=batchSize, shuffle=True)
    validationDataLoader = torch.utils.data.DataLoader(validation, batch_size=batchSize, shuffle=True)

    # 初始化模型并转移到GPU上
    netModel = NetM32()
    # netModel = NetM20()
    # netModel = NetM10()
    if torch.cuda.is_available():
        netModel.cuda()

    # ==================================== 训练模型 ====================================
    # =================================================================================
    # 定义损失函数（loss criterion）
    criterion = nn.CrossEntropyLoss()
    # 创建随机梯度下降优化器（optimizer）
    optimizer = torch.optim.SGD(netModel.parameters(), lr=0.001, momentum=0.9)

    print("Training and validation sets loaded.")
    print("Training Start: " + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    print("Model Training...")
    epochs = 30
    training_loss_list, training_correct_list, val_loss_list, val_acc_list = [], [], [], []

    # for epoch in tqdm(range(epochs), desc="Training", unit="Epoch"):
    for epoch in range(epochs):
        # =============== 训练模型 ===============
        trainingLoss, trainingCorrect = 0.0, 0.0
        for i, (inputs, bearingLabels) in enumerate(trainingDataLoader, 0):
            if torch.cuda.is_available():
                inputs, bearingLabels = inputs.cuda(), bearingLabels.cuda()

            bearingLabels = torch.squeeze(bearingLabels, 1)  # 降维，可删除，仅当前模型使用

            optimizer.zero_grad()  # 初始化参数grad的值
            outputBearing = netModel(inputs)
            loss = criterion(outputBearing, bearingLabels)  # 使用交叉熵计算损失
            loss.backward()  # 反向传播
            optimizer.step()  # 更新随机梯度下降参数
            trainingLoss += loss.item()  # runningLoss用于统计loss平均值
            trainingCorrect += (outputBearing.argmax(1) == bearingLabels.argmax(1)).type(torch.float).sum().item()
        training_loss = trainingLoss / len(trainingDataLoader)
        training_correct = trainingCorrect / len(trainingDataLoader.dataset)

        # =============== 验证集验证每一个epoch模型 ===============
        valLoss, correct = 0.0, 0.0
        netModel.eval()
        with torch.no_grad():
            for i_valid, (inputs_v, bearingLabels_v) in enumerate(validationDataLoader,0):
                if torch.cuda.is_available():
                    inputs_v, bearingLabels_v = inputs_v.cuda(), bearingLabels_v.cuda()
                outputBearing_v = netModel(inputs_v)  # 拟合输出
                bearingLabels_v = torch.squeeze(bearingLabels_v, 1)  # 降维，可删除，仅当前模型使用
                loss_v = criterion(outputBearing_v, bearingLabels_v)
                valLoss += loss_v.item()
                correct += (outputBearing_v.argmax(1) == bearingLabels_v.argmax(1)).type(torch.float).sum().item()
        # 每一个epoch下的模型的损失和精确度
        val_loss = valLoss / len(validationDataLoader)
        val_acc = correct / len(validationDataLoader.dataset)
        
        # 列表存放训练损失的损失值和正确率、验证损失的损失值和正确率
        training_loss_list.append(training_loss)
        training_correct_list.append(training_correct)
        val_loss_list.append(val_loss)
        val_acc_list.append(val_acc)

        print('Epoch %d, Running Loss: %.3f, Correct = %f. Accuracy of the network on the validation set: %.3f %%' % (epoch + 1, training_loss, training_correct, (val_acc*100)))
    
    # print训练结束时间
    TrainingEndedTime = time.strftime("%Y%m%d%H%M", time.localtime())
    savedPath = getpath("netModelSavedPath")
    print("Training && Validation Ended: " + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

    # ==================================== 保存参数 ====================================
    # =================================================================================
    if val_acc >= 0.5:
        # 保存网络模型参数
        savedName = savedPath + "\\" + TrainingEndedTime + "_model.pth"
        torch.save(netModel.state_dict(), savedName)
        # 保存损失、平均损失以及正确率
        savedParam = savedPath + "\\" + TrainingEndedTime + "_mParam.mat"
        savemat(savedParam,
                {"TrainingLoss": training_loss_list, "TrainingAccuracy ": training_correct_list, 
                "ValidationLoss": val_loss_list, "ValidationAccuracy": val_acc_list, "correctRate": val_acc})
        print("Model saved at: " + savedName)
    else:
        print("Low model accuracy, not saved.")

    savePicName = savedPath + "\\" + TrainingEndedTime + "_mPic.jpg"
    
    plt.plot(list(range(len(training_loss_list))), training_loss_list, label="training loss")
    plt.plot(list(range(len(training_correct_list))), training_correct_list, label="training accuracy")
    plt.plot(list(range(len(val_loss_list))), val_loss_list, label="val loss")
    plt.plot(list(range(len(val_acc_list))), val_acc_list, label="val accuracy")
    plt.legend()
    plt.savefig(savePicName, dpi=240)
    # plt.show()

    print("================== プログラム終了 ==================")
