# -*- coding: utf-8 -*-
"""
    @ProjectName: CBF
    @File: test.py
    @Author: Chaos
    @Date: 2023/5/22
    @Description: 测试文件
"""
from ConvBeamForming import mainCBF
from scipy.io import loadmat
import matplotlib.pyplot as plt
import time


if __name__ == '__main__':
    print("================== 常规波束形成计算部分 ==================")
    startTime = time.time()
    # 加载数据
    savedName = "..\\dataraw\\sdata_M20theta25snr15.mat"
    signalDict = loadmat(savedName)
    signalDict['fs'] = 51200  # 由于原始数据中没有 fs 的 key 值，测试时填上

    # # 滑动时间窗长度，可根据实际需要设定
    # T = 0.2  # 信号周期
    # interval = 1
    # fs = 51200  # 水平阵采样频率
    # winLen = int(fs * (interval + T / 2))
    # overlapLen = int(0.1 * fs)

    # 计算数据
    _, _, fig = mainCBF(signalDict)
    fig.savefig("./1.jpg")

    endTime = time.time()
    t_sum = endTime - startTime
    print("运行时间：" + str(t_sum))
    print("================== プログラム終了 ==================")
