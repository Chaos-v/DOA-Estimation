# -*- coding: utf-8 -*-
"""
    @File: transferLearning.py
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
    print("================== Transfer Learning ==================")
    print("Program Start Time at: " + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

    # ==================================== 导入数据集 ====================================
    # datasetPath_val_1 = ".\\dataset\\M32tmp\\M32SNR10"
    # datasetPath_val_2 = ".\\dataset\\M32tmp\\M32SNR0"
    # datasetPath_val_3 = ".\\dataset\\M32tmp\\M32SNR-5"
    # datasetPath_val_4 = ".\\dataset\\M32tmp\\M32SNR-10"
    # 总的数据集保存位置
    subFolderPath = ".\\dataset\\M32tmp"

    # 初始化模型并转移到GPU上
    modelPath = ".\\CNNmodel\\model\\M32_SNR20_model.pth"
    netModel = NetM32()
    # netModel = NetM20()
    # netModel = NetM10()

    # with open(os.path.join(datasetPath_val_1, "trainLoader.pkl"), 'rb') as f:
    #         d1_tmp = pickle.load(f)
    # with open(os.path.join(datasetPath_val_1, "validation.pkl"), 'rb') as f:
    #         v1_tmp = pickle.load(f)
    # with open(os.path.join(datasetPath_val_1, "testLoader.pkl"), 'rb') as f:
    #         t1_tmp = pickle.load(f)
    
    # with open(os.path.join(datasetPath_val_2, "trainLoader.pkl"), 'rb') as f:
    #         d2_tmp = pickle.load(f)
    # with open(os.path.join(datasetPath_val_2, "validation.pkl"), 'rb') as f:
    #         v2_tmp = pickle.load(f)
    # with open(os.path.join(datasetPath_val_2, "testLoader.pkl"), 'rb') as f:
    #         t2_tmp = pickle.load(f)
   
    # with open(os.path.join(datasetPath_val_3, "trainLoader.pkl"), 'rb') as f:
    #         d3_tmp = pickle.load(f)
    # with open(os.path.join(datasetPath_val_3, "validation.pkl"), 'rb') as f:
    #         v3_tmp = pickle.load(f)
    # with open(os.path.join(datasetPath_val_3, "testLoader.pkl"), 'rb') as f:
    #         t3_tmp = pickle.load(f)

    # with open(os.path.join(datasetPath_val_4, "trainLoader.pkl"), 'rb') as f:
    #         d4_tmp = pickle.load(f)
    # with open(os.path.join(datasetPath_val_4, "validation.pkl"), 'rb') as f:
    #         v4_tmp = pickle.load(f)
    # with open(os.path.join(datasetPath_val_4, "testLoader.pkl"), 'rb') as f:
    #         t4_tmp = pickle.load(f)
    
    # training = d1_tmp + d2_tmp + d3_tmp + d4_tmp
    # validate = v1_tmp + v2_tmp + v3_tmp + v4_tmp
    # test = t1_tmp + t2_tmp + t3_tmp + t4_tmp

    # print(str(len(d1_tmp)), str(len(d2_tmp)), str(len(d3_tmp)), str(len(d4_tmp)), )

    # # 在本文件使用时需要注释下面保存语句
    # if not os.path.isdir(subFolderPath):
    #     os.makedirs(subFolderPath)
    # with open(os.path.join(subFolderPath, "trainLoader.pkl"), 'wb') as f:
    #     pickle.dump(training, f)
    # with open(os.path.join(subFolderPath, "validation.pkl"), 'wb') as f:
    #     pickle.dump(validate, f)
    # with open(os.path.join(subFolderPath, "testLoader.pkl"), 'wb') as f:
    #     pickle.dump(test, f)

    # 加载已经保存好的数据集
    with open(os.path.join(subFolderPath, "trainLoader.pkl"), 'rb') as f:
            trainLoader = pickle.load(f)
    with open(os.path.join(subFolderPath, "validation.pkl"), 'rb') as f:
            validation = pickle.load(f)

    print("训练集大小：" + str(len(trainLoader)))

    batchSize = 32
    trainingDataLoader = torch.utils.data.DataLoader(trainLoader, batch_size=batchSize, shuffle=True)
    validationDataLoader = torch.utils.data.DataLoader(validation, batch_size=batchSize, shuffle=True)

    netModel.load_state_dict(torch.load(modelPath))
    if torch.cuda.is_available():
        netModel.cuda()

    # ==================================== 训练模型 ====================================
    # =================================================================================
    # 定义损失函数（loss criterion）
    criterion = nn.CrossEntropyLoss()
    # 创建随机梯度下降优化器（optimizer）
    optimizer = torch.optim.SGD(netModel.parameters(), lr=0.0005, momentum=0.9)

    print("Training and validation sets loaded.")
    print("Training Start: " + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    print("Model Training...")
    epochs = 20
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
        training_loss = trainingLoss/len(trainingDataLoader)
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

        print('Epoch %d, Running Loss: %.3f, Correct = %f. Accuracy of the network on the validation set: %d %%' % (epoch + 1, training_loss, training_correct, (val_acc*100)))
    
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
        print("Model saved.")
    else:
        print("Low model accuracy, not saved.")

    savePicName = savedPath + "\\" + TrainingEndedTime + "_mPic.jpg"
    
    plt.plot(list(range(len(training_loss_list))), training_loss_list, label="training loss")
    plt.plot(list(range(len(training_correct_list))), training_correct_list, label="training accuracy")
    plt.plot(list(range(len(val_loss_list))), val_loss_list, label="val loss")
    plt.plot(list(range(len(val_acc_list))), val_acc_list, label="val accuracy")
    plt.legend()
    plt.savefig(savePicName, dpi=240)

    print("================== プログラム終了 ==================")
