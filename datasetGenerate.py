# -*- coding: utf-8 -*-
"""
    @ProjectName: DecisionAlg
    @File: datasetGenerate.py
    @Author: Chaos
    @Date: 2023/5/24
    @Description: 用于生成决策算法所需要的数据集，包括相关的函数
"""
import numpy as np
from scipy.signal import hilbert
import pickle
import os
import time
import torch


def signalGenerate(theta0, sampleTime=10, eleNum=10, eleSpacing=1.5,
                   sampFreq=51200, freq0=50, sigPeriod=0.2, sigInterval=1, sigAmp=1, SNR=0):
    """
    针对线列阵的 CW 信号生成函数。
    生成具有一定方向信息的阵列采集信息。使用前需要根据情况做 Hilbert 变换。

    :param theta0: 入射角度，单位：°
    :param sampleTime: 阵列采集时间，也就是每一个换能器的时间长度，单位：s
    :param eleNum: 阵元数量
    :param eleSpacing: 阵元间距，单位：m
    :param sampFreq: 采样频率
    :param freq0: 发射信号频率，单位：Hz
    :param sigPeriod: 发射信号周期，单位：s
    :param sigInterval: 发射间隔，单位：s
    :param sigAmp: 发射信号幅度
    :param SNR: 信噪比
    :return: arraySignal，tScale
    """
    c = 1500  # 声速
    # 将入射角度转化为弧度
    theta0 = np.deg2rad(theta0)
    tou0 = eleSpacing / c * np.sin(theta0)  # 两相邻接收器接收声压的时间差

    # 生成发射信号
    t1 = np.arange(0, sigPeriod, 1 / sampFreq)
    transmitSignal = sigAmp * np.sin(2 * np.pi * freq0 * t1)  # 发射信号部分

    # 根据输入的信噪比计算需要添加的噪声均值(mean)和方差(var)
    meanNoise, varNoise = 0, (np.sqrt(np.sum(transmitSignal ** 2) / len(transmitSignal)) / (10 ** (SNR / 10)))

    N = int(sampFreq * sampleTime)  # 每个水听器采集点数
    tScale = np.arange(0, sampleTime, 1 / sampFreq)

    arraySignal = np.zeros((eleNum, N), dtype=complex)  # 初始化阵列，全0
    for index in range(eleNum):
        noise = np.random.normal(meanNoise, varNoise, N)  # 每循环一次先生成对应水听器的噪声
        # noise = np.zeros(int(N))
        tmpTou = index * tou0  # 当前水听器距离0号水听器的接收声压的时间差
        tmpFullSig = np.concatenate((np.zeros(int(sigInterval / 2 * sampFreq + np.ceil(tmpTou * sampFreq))),
                                     transmitSignal, np.zeros(int(sigInterval / 2 * sampFreq - np.ceil(tmpTou * sampFreq)))))
        tmpNCycle = int(np.ceil(N / len(tmpFullSig)))
        tmpEleData = np.tile(tmpFullSig, tmpNCycle)[:N] + noise
        tmpEleData = hilbert(tmpEleData)
        arraySignal[index, :] = tmpEleData

    return arraySignal, tScale


def sigCW(theta0, sampleTime=2, eleSpacing=1.5, eleNum=10, sampleFreq=51200, freq0=50, sigAmp=1, SNR=0):
    """
    生成无占空比的，连续的水平阵列采样信号

    :param theta0: 入射角度，单位：°
    :param sampleTime: 阵列采集时间，也就是每一个换能器的时间长度，单位：s
    :param eleNum: 阵元数量
    :param eleSpacing: 阵元间距，单位：m
    :param sampleFreq: 采样频率
    :param freq0: 发射信号频率，单位：Hz
    :param sigAmp: 发射信号幅度
    :param SNR: 信噪比
    :return: arraySignal，tScale
    """
    c = 1500
    theta0 = np.deg2rad(theta0)
    tou0 = eleSpacing / c * np.sin(theta0)  # 时间差
    phi0 = 2 * np.pi * freq0 * tou0

    # 生成发射信号计算添加噪声的均值方差
    N = int(sampleFreq * sampleTime)
    tScale = np.arange(0, sampleTime, 1 / sampleFreq)
    # 根据输入的信噪比计算需要添加的噪声均值(mean)和方差(var)
    transmitSignal = sigAmp * np.sin(2 * np.pi * freq0 * tScale)  # 标准发射信号部分
    meanNoise, varNoise = 0, (np.sqrt(np.sum(transmitSignal ** 2) / len(transmitSignal)) / (10 ** (SNR / 10)))

    arraySignal = np.zeros((eleNum, N), dtype=complex)
    for index in range(eleNum):
        noise = np.random.normal(meanNoise, varNoise, N)
        tmpSig = sigAmp * np.sin(2 * np.pi * freq0 * tScale - index * phi0)  # 发射信号部分
        arraySignal[index, :] = hilbert(tmpSig + noise)  # hilbert变换
        # arraySignal[index, :] = tmpSig + noise

    return arraySignal, tScale


def dataSplit(fpath):
    """加载同时预处理数据集，只针对特定数据生效，并且做一个分割，分割成训练集，验证集，测试集并保存"""
    with open(fpath, 'rb') as f:
        data = pickle.load(f)

    # ==================================== 数据集预处理 ====================================
    dataset = [None] * len(data)
    tmpData = np.zeros((2, 10, 10), dtype=float)
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
    print("================== datasetGenerate ==================")
    print("Start Time: " + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    # 论文需求需要改变的参数
    thetaList = list(range(-90, 91, 1))
    snrList = [20, 10, 0, -10]
    M = 10  # 阵元数量

    # 固定不变参数
    recvTime = 2  # 信号接收时间
    d = 1.5  # 阵元间距
    fs = 51200  # 水平阵采样频率
    N = recvTime * fs  # 采样点数

    f0_list = [500]  # 取500是满足阵元间距等于半波长的条件
    f0 = f0_list[np.random.randint(0, len(f0_list))]  # 发射信号频率

    sampTimes = 1000  # 每个角度需要的采样次数
    lenList = int(sampTimes * len(thetaList))

    # 创建文件存放目录
    datasetFolder = 'dataset'
    if not os.path.exists(datasetFolder):
        os.mkdir(os.path.join(datasetFolder))

    print("Dataset Generating... Do not close the VSCode.")

    for snr in snrList:
        print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + "\tSNR = " + str(snr))
        # 不同信噪比下创建数据集
        datasetList = [None] * lenList
        num = 0
        for thetaIndex in range(len(thetaList)):
            theta0 = thetaList[thetaIndex]
            # 不同入射角
            for i in range(sampTimes):
                # 每次循环生成一个随机的幅度以及发射信号频率
                amp = (np.random.randint(0, 10) + 1) / 10  # 信号幅度，离散均匀分布取值0.1-1.0的信号幅度
                sigTmp, _ = sigCW(theta0, sampleTime=recvTime, eleSpacing=d, eleNum=M, sampleFreq=fs, freq0=f0, sigAmp=amp, SNR=snr)
                R_matrix = sigTmp @ sigTmp.T.conjugate() / N
                datasetList[num] = (R_matrix, theta0, amp)
                num += 1

        fileName = "M" + str(M) + "SNR" + str(snr) + ".pkl"
        filePath = os.path.join(datasetFolder, fileName)
        with open(filePath, 'wb') as f:
            pickle.dump(datasetList, f)

    # 读取文件
    # with open('snr=10.pkl', 'rb') as f:
    #     data = pickle.load(f)
    print("End Time: " + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    print("================== プログラム終了 ==================")
