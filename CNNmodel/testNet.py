# -*- coding: utf-8 -*-
"""
    @File: testNet.py
    @Author: Chaos
    @Date: 2023/6/19
    @Description: 
"""
from scipy.io import loadmat
import matplotlib.pyplot as plt
from DOAEstimation import CNN_DOAEstimation
from NetModel import NetM20
import time


if __name__ == '__main__':
    print("================== testNet ==================")
    startTime = time.time()
    # 输入数据加载
    datapath = "..\\dataraw\\sdata_M20theta25snr15.mat"
    signalDict = loadmat(datapath)

    # winLen, overlapLen = 56320, 5120  # 设置滑动窗长度以及重叠部分
    _, _, fig = CNN_DOAEstimation(signalDict)  # 7输入指定参数调用即可
    fig.savefig("./1.jpg")

    endTime = time.time()
    t_sum = endTime - startTime
    print("运行时间：" + str(t_sum))
    print("================== プログラム終了 ==================")
