# -*- coding: utf-8 -*-
"""
    @ProjectName: DecisionAlg
    @File: datasetGenerate.py
    @Author: Chaos
    @Date: 2023/5/24
    @Description: 用于生成决策算法所需要的数据集，包括相关的函数
"""
from ConvBeamForming import CBF
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert


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


def sigCW(theta0, sampleTime=2, eleSpacing=1.5, eleNum=10,
                  sampFreq=51200, freq0=50, sigAmp=1, SNR=0):
    """
    生成无占空比的，连续的水平阵列采样信号

    :param theta0: 入射角度，单位：°
    :param sampleTime: 阵列采集时间，也就是每一个换能器的时间长度，单位：s
    :param eleNum: 阵元数量
    :param eleSpacing: 阵元间距，单位：m
    :param sampFreq: 采样频率
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
    N = int(sampFreq * sampleTime)
    tScale = np.arange(0, sampleTime, 1 / sampFreq)
    # 根据输入的信噪比计算需要添加的噪声均值(mean)和方差(var)
    transmitSignal = sigAmp * np.sin(2 * np.pi * freq0 * tScale)  # 标准发射信号部分
    meanNoise, varNoise = 0, (np.sqrt(np.sum(transmitSignal ** 2) / len(transmitSignal)) / (10 ** (SNR / 10)))

    arraySignal = np.zeros((eleNum, N), dtype=complex)
    for index in range(eleNum):
        noise = np.random.normal(meanNoise, varNoise, N)
        tmpSig = sigAmp * np.sin(2 * np.pi * freq0 * tScale - index * phi0)  # 发射信号部分
        # arraySignal[index, :] = hilbert(tmpSig + noise)
        arraySignal[index, :] = tmpSig + noise

    return arraySignal, tScale


if __name__ == '__main__':
    print("================== datasetGenerate ==================")
    # 接收阵部分
    theta0 = 10  # 入射角度

    recevTime = 2  # 信号接收时间
    M = 10  # 阵元数量
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
    cwSignal, tScale = sigCW(theta0, sampleTime=recevTime, eleNum=M, eleSpacing=d,
                             sampFreq=fs, freq0=f0, sigAmp=amp, SNR=snr)

    # # ==================== 滑动窗截取 ====================
    # winLen = interval + T / 2  # 滑动时间窗长度
    # overlapLen = int(0.1 * fs)
    # shape = (np.size(cwSignal, 0), int(fs * winLen))
    # v = np.lib.stride_tricks.sliding_window_view(cwSignal, shape)[:, ::overlapLen, :].squeeze()
    #
    # # 每一个窗做一个 CBF
    # for i in range(len(v)):
    #     if i == 0:
    #         theta_Deg, P_dB = CBF(v[i], freq0=f0, delta=d)
    #     else:
    #         theta_Deg, tmpP = CBF(v[i], freq0=f0, delta=d)
    #         P_dB = np.vstack([P_dB, tmpP])

    theta_Deg, P_dB = CBF(cwSignal, f0, d)

    fig = plt.figure()
    plt.plot(theta_Deg, P_dB)
    # plt.pcolor(theta_Deg, np.linspace(0, len(v)-1, len(v)), P_dB)
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    # plt.colorbar()
    plt.xlabel('angle')
    plt.show()

    print("================== プログラム終了 ==================")
