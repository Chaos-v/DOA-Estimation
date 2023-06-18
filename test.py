# -*- coding: utf-8 -*-
"""
    @ProjectName: DecisionAlg
    @File: test.py
    @Author: Chaos
    @Date: 2023/5/22
    @Description: 
"""
from ConvBeamForming import CBF
from datasetGenerate import *
import matplotlib.pyplot as plt


if __name__ == '__main__':
    print("================== test ==================")
    # dataPath = "./Signal.mat"
    # data = io.loadmat(dataPath)
    # signalData = data['S_r']
    # thetaDeg, PdB = CBF(signalData, 62.5, 12)
    #
    # fig1 = plt.figure(1)
    # plt.plot(thetaDeg, PdB)
    # plt.show()

    # 接收阵部分
    theta0 = 25
    recevTime = 10  # 信号接收时间
    M = 32  # 阵元数量
    d = 1.5  # 阵元间距
    fs = 51200  # 水平阵采样频率
    # c = 1500  # 声速

    # 发射部分
    f0 = 300  # 发射信号频率
    T = 0.2  # 信号周期
    interval = 1
    amp = 1  # 信号幅度
    snr = 5

    # ==================== 产生仿真信号 ====================
    cwSignal, tScale = signalGenerate(theta0, sampleTime=recevTime, eleNum=M, eleSpacing=d, sampFreq=fs, freq0=f0,
                                      sigPeriod=T, sigInterval=interval, sigAmp=amp, SNR=snr)

    # ==================== 滑动窗截取 ====================
    winLen = interval + T / 2  # 滑动时间窗长度
    overlapLen = int(0.1 * fs)
    shape = (np.size(cwSignal, 0), int(fs * winLen))
    v = np.lib.stride_tricks.sliding_window_view(cwSignal, shape)[:, ::overlapLen, :].squeeze()

    # 每一个窗做一个 CBF
    for i in range(len(v)):
        if i == 0:
            theta_Deg, P_dB = CBF(v[i], freq0=f0, delta=d)
        else:
            theta_Deg, tmpP = CBF(v[i], freq0=f0, delta=d)
            P_dB = np.vstack([P_dB, tmpP])

    fig = plt.figure()
    # plt.plot(theta_Deg, P_dB)
    plt.pcolor(theta_Deg, np.linspace(0, len(v) - 1, len(v)), P_dB)
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.colorbar()
    plt.xlabel('angle')
    plt.show()

    # plt.close(fig=1)
    print("================== プログラム終了 ==================")
