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


if __name__ == '__main__':
    print("================== 常规波束形成计算部分 ==================")
    # 加载数据
    savedName = "..\\dataraw\\sdata_theta25snr20.mat"
    matdata = loadmat(savedName)
    signal, f0 = matdata["signal"], float(matdata["f0"])

    # 滑动时间窗长度，可根据实际需要设定
    T = 0.2  # 信号周期
    interval = 1
    fs = 51200  # 水平阵采样频率
    winLen = int(fs * (interval + T / 2))
    overlapLen = int(0.1 * fs)

    # 计算数据
    _, _, fig = mainCBF(signal, f0, winLen, overlapLen)
    fig.savefig("./1.jpg")

    print("================== プログラム終了 ==================")
