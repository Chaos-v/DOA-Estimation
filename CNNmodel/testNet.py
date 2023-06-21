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
from NetModel import NetM10


if __name__ == '__main__':
    print("================== testNet ==================")
    # 输入数据加载
    datapath = "..\\dataraw\\sdata_theta-25snr20.mat"
    signal = loadmat(datapath)["signal"]

    winLen, overlapLen = 56320, 5120  # 设置滑动窗长度以及重叠部分
    _, _, fig = CNN_DOAEstimation(signal, winLen, overlapLen)  # 输入指定参数调用即可
    fig.savefig("./1.jpg")

    print("================== プログラム終了 ==================")
