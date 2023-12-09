"""
    @File: datasetSplit.py
    @Author: Chaos
    @Date: 2023/8/14
    @Description: 分割数据集
"""
import os
import pickle
import numpy as np
import torch.cuda
import torch.nn as nn


def dataSplit(fpath):
    """加载同时预处理数据集，只针对特定数据生效，并且做一个分割，分割成训练集，验证集，测试集并保存"""
    with open(fpath, 'rb') as f:
        data = pickle.load(f)

    # ==================================== 数据集预处理 ====================================
    dataset = [None] * len(data)
    tmpData = np.zeros((2, 32, 32), dtype=float)
    # tmpData = np.zeros((2, 20, 20), dtype=float)
    # tmpData = np.zeros((2, 10, 10), dtype=float)

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
    # 在本文件使用时需要注释下面保存语句
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
    print("================== datasetSplit ==================")
    datasetFolder = ".\\dataset"
    datasetFiles = os.listdir(datasetFolder)
    print(datasetFiles)

    trainLoader, validation, testLoader = [], [], []
    for file in datasetFiles:
        path = os.path.join(datasetFolder,file)
        trainingTmp, validataTmp, testTmp = dataSplit(path)
        trainLoader.extend(trainingTmp)
        validation.extend(validataTmp)
        testLoader.extend(testTmp)
        print(len(trainLoader))

    with open(os.path.join(datasetFolder, "trainLoader.pkl"), 'wb') as f:
        pickle.dump(trainLoader, f)
    with open(os.path.join(datasetFolder, "validation.pkl"), 'wb') as f:
        pickle.dump(validation, f)
    with open(os.path.join(datasetFolder, "testLoader.pkl"), 'wb') as f:
        pickle.dump(testLoader, f)
    
    print("================== プログラム終了 ==================")