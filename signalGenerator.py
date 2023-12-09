# -*- coding: utf-8 -*-
"""
    @File: signalGenerator.py
    @Author: Chaos
    @Date: 2023/6/21
    @Description: 生成各式各样的仿真信号
"""
import numpy as np
from scipy.signal import hilbert
from numpy.random import uniform


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
    c = uniform(low=1450, high=1550)  # 声速
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

    # arraySignal = np.zeros((eleNum, N), dtype=complex)  # 初始化阵列，全0
    arraySignal = np.zeros((eleNum, N), dtype=float)
    for index in range(eleNum):
        noise = np.random.normal(meanNoise, varNoise, N)  # 每循环一次先生成对应水听器的噪声
        # noise = np.zeros(int(N))
        tmpTou = index * tou0  # 当前水听器距离0号水听器的接收声压的时间差
        tmpFullSig = np.concatenate((np.zeros(int(sigInterval / 2 * sampFreq + np.ceil(tmpTou * sampFreq))),
                                     transmitSignal, np.zeros(int(sigInterval / 2 * sampFreq - np.ceil(tmpTou * sampFreq)))))
        tmpNCycle = int(np.ceil(N / len(tmpFullSig)))
        tmpEleData = np.tile(tmpFullSig, tmpNCycle)[:N] + noise
        # tmpEleData = hilbert(tmpEleData)
        arraySignal[index, :] = tmpEleData

    return arraySignal, tScale


def sigCW(theta0, sampleTime=2, eleSpacing=1.5, eleNum=10, sampleFreq=51200, freq0=50, sigAmp=1, SNR=0, hil=False):
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
    :param hil: 是否希尔伯特变换，bool，默认false
    :return: arraySignal，tScale
    """
    # c = uniform(low=1450, high=1550)  # 声速
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

    if hil is False:
        arraySignal = np.zeros((eleNum, N), dtype=float)
        for index in range(eleNum):
            noise = np.random.normal(meanNoise, varNoise, N)
            tmpSig = sigAmp * np.sin(2 * np.pi * freq0 * tScale - index * phi0)  # 发射信号部分
            arraySignal[index, :] = tmpSig + noise
    else:
        arraySignal = np.zeros((eleNum, N), dtype=complex)
        for index in range(eleNum):
            noise = np.random.normal(meanNoise, varNoise, N)
            tmpSig = sigAmp * np.sin(2 * np.pi * freq0 * tScale - index * phi0)  # 发射信号部分
            arraySignal[index, :] = hilbert(tmpSig + noise)  # hilbert变换

    return arraySignal, tScale


if __name__ == '__main__':
    print("================== signalGenerator ==================")
