# -*- coding: utf-8 -*-
"""
    @File: datarawGenerator.py
    @Author: Chaos
    @Date: 2023/6/19
    @Description: 生成要求的模拟实测信号并保存为 mat 文件存放于 dataraw 文件夹下
"""
from scipy.io import savemat
from signalGenerator import sigCW


if __name__ == '__main__':
    print("================== signalGenerator ==================")
    # ==================== 参数部分 ====================
    theta0 = 12
    recevTime = 10  # 信号接收时间
    M = 10  # 阵元数量
    d = 1.5  # 阵元间距
    fs = 51200  # 水平阵采样频率
    # c = 1500  # 声速

    # 发射部分
    f0 = 500  # 发射信号频率
    T = 0.2  # 信号周期
    interval = 1
    amp = 1  # 信号幅度
    snr = 20

    # ==================== 产生仿真信号 ====================
    signal, tScale = sigCW(theta0, sampleTime=recevTime, eleNum=M, eleSpacing=d, sampleFreq=fs, freq0=f0, sigAmp=amp, SNR=snr)

    # ==================== 保存数据部分 ====================
    savedPath = ".\\dataraw\\sdata_theta" + str(theta0) + "snr" + str(snr) + ".mat"
    savemat(savedPath, {"signal": signal, "f0": f0, "theta0": theta0})

    print("================== プログラム終了 ==================")
