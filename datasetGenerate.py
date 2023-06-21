# -*- coding: utf-8 -*-
"""
    @ProjectName: DecisionAlg
    @File: datasetGenerate.py
    @Author: Chaos
    @Date: 2023/5/24
    @Description: 用于生成卷积神经网络算法所需要的数据集，包括相关的函数
"""
import numpy as np
from scipy.signal import hilbert
import pickle
import os
import time
from signalGenerator import sigCW


if __name__ == '__main__':
    print("================== datasetGenerate ==================")
    print("Start Time: " + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    # 论文需求需要改变的参数
    thetaList = list(range(-90, 91, 1))
    snrList = [20, 10, 0, -10]
    M = 20  # 阵元数量

    # 固定不变参数
    recvTime = 1  # 信号接收时间
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
