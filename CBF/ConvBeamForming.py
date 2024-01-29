# -*- coding: utf-8 -*-
"""
    @ProjectName: CBF \n
    @File: ConvBeamForming.py \n
    @Author: Chaos \n
    @Date: 2023/5/18 \n
    @Usage:
        1. 封装在软件前，根据实际阵列信息对 CBFCore 函数中的阵元间距进行设置；
        2. 根据实际需要设置滑动窗 (slideWinView) 函数的窗口长度以及窗口重叠部分的长度
        3. 直接调用入口函数，按照要求给定参数即可。 \n
    @Description: 常规波束形成（仿真），算法部分位于 CBFCore 的函数中。
                  原理介绍：
                        1.截取阵列采集的一段时间数据并且做 Hilbert 变换；
                        2.设置滑动时间窗截取信号调用 CBF 算法进行计算；
                        3.生成方位时间图
"""
import numpy as np
from scipy.signal import hilbert
from scipy.fftpack import fft
import matplotlib.pyplot as plt


def mainCBF(sigDict: dict, win=51200, overlap=51200):
    """
    完整CBF算法的封装，用于软件内部调用，使用前提是已知相关参数。

    封装时需要更改:
    (1) 滑动窗函数 (slideWinView) 的输入变量 win，overlap 的默认值；
    (2) 常规波束形成算法 (CBF) 函数的输入变量 freq0, delta 的默认值。
    (3) 预处理函数 (CBFPreProcess) 中阵列的默认采样频率 fs 的默认值，如果 mainCBF 方法中传入的参量中含有采样频率的key值，则忽略该条
    其中，变量 win, overlap, freq0 的默认值可直接在 mainCBF 函数中修改，修改后直接传递到对应函数。
    变量 delta 的值需要单独在 CBFCore 函数中进行修改。

    :param sigDict: 包含阵列信号的字典变量，其中阵列信号部分中：阵元 × 采集信号（行 × 列）
    :param win: 表示滑动窗长度的快拍数。Len = 采样频率 × 窗口时间长度。
    :param overlap: 相邻两个滑动窗重叠的快拍数。Len = 采样频率 × 重叠部分的时间长度
    :return: P_dB, theta_Deg, fig
    """
    # 阵列信号预处理
    sigPreprocess, f0 = CBFPreProcess(sigDict)
    # 获取滑动窗
    v = slideWinView(sigPreprocess, win, overlap)
    dim0, _, _ = v.shape

    # 每一个窗做一个 CBF
    for i in range(len(v)):
        if i == 0:
            theta_Deg, P_dB = CBFCore(v[i], freq0=f0)
        else:
            theta_Deg, tmpP = CBFCore(v[i], freq0=f0)
            P_dB = np.vstack([P_dB, tmpP])

    fig = plt.figure()
    plt.pcolor(theta_Deg, np.linspace(0, len(v) - 1, len(v)), P_dB)
    plt.ylim(0, dim0-1)
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.colorbar()
    plt.xlabel('角度（°）')
    plt.ylabel('时间窗')

    return P_dB, theta_Deg, fig


def CBFPreProcess(sigDict):
    """
    针对传入的实际采集的阵列信号进行预处理，预处理方法根据实际需要添加在该函数中，最终输出预处理后的阵列信号变量。
    该函数可以根据实际需要进行调用。
    :param sigDict: 阵列信号，阵元 × 采集信号（行 × 列）
    :return: arraySig
    """
    # 提取需要信息
    sig = sigDict["signal"]

    if 'f0' in sigDict.keys():  # 获取发射信号的可能频率
        f0 = float(sigDict["f0"])
    else:
        if 'fs' in sigDict.keys():
            fs = sigDict["fs"]
            f0 = getFreq(sig, fs)
        else:
            f0 = getFreq(sig, fs=51200)  # 这块而自己设置

    m = np.size(sig, 0)  # 输入信号矩阵的行数，也即阵元的数量
    n = np.size(sig, 1)  # 输入信号矩阵的列数

    arraySig = np.zeros((m, n), dtype=complex)
    for i in range(m):
        # Hilbert 变换
        tmp = sig[i, :]
        arraySig[i, :] = hilbert(tmp)

    return arraySig, f0


def getFreq(sig, fs=51200):
    """
    获取信号的可能频率，其实就是个简单的fft
    :param sig: 阵列信号，阵元 × 采集信号（行 × 列）
    :param fs:
    :return:
    """
    freq = 0
    tmpSig = sig[0, :]
    L = len(tmpSig)  # Length of signal
    # 计算信号的傅里叶变换，计算双侧频谱 P2。然后基于 P2 和偶数信号长度 L 计算单侧频谱 P1。
    P2 = abs(fft(tmpSig) / L)
    P1 = P2[: int(L/2+1)]
    P1[1: -1] = 2 * P1[1: -1]
    # f_list = fs * np.linspace(start=0, stop=L/2, num=int(L/2+1)) / L
    freq = float(fs * np.argmax(P1) / L)
    return freq


def slideWinView(sig, win, overlap):
    """
    滑动窗函数

    :param sig: 阵列信号，阵元 × 采集信号（行 × 列）
    :param win: 表示滑动窗长度的快拍数。Len = 采样频率 × 窗口时间长度。
    :param overlap: 相邻两个滑动窗重叠的快拍数。Len = 采样频率 × 重叠部分的时间长度
    :return: v: 包含每一个窗口数据的三维矩阵
    """
    shape = (np.size(sig, 0), int(win))
    v = np.lib.stride_tricks.sliding_window_view(sig, shape)[:, ::overlap, :].squeeze()
    return v


def CBFCore(aSignal, freq0, delta=1.5):
    """
    常规波束形成算法

    阵元编号(N个阵元)：
        o- - - -o- - - -o- ... -o- - - -o  \n
        0- - - -1- - - -2- ... -N-2- - - -N-1  \n

    :param aSignal: 阵列信号，阵元 × 采集信号，需要再函数调用前对采集的信号做 Hilbert 变换
    :param freq0: 发射信号频率
    :param delta: 阵元间距，不写就是默认1.5m，实际封装在软件中时使用默认值。

    :return: theta: 扫描角度的向量
             PdB: 常规波束形成器的输出功率
    """
    M, N = aSignal.shape[0], aSignal.shape[1]  # 阵元数量M，采样点数N
    R_hat = aSignal @ aSignal.T.conjugate() / N  # 基阵的采样协方差矩阵

    thetaDeg = np.linspace(-90, 90, 181)
    thetaRad = np.deg2rad(thetaDeg)  # 设定扫描角度以及扫描间隔，弧度制的向量
    L_thetaArray = len(thetaRad)

    # 计算每一个角度对应的加权向量，每一行对应一个角度
    # 同时计算每一个角度的常规波束形成器的输出功率
    weight = np.zeros((L_thetaArray, M), dtype=complex)  # 每一个扫描角度的加权向量，加权向量Omega是扫描角度的函数，与时间无关
    P = np.zeros(L_thetaArray, dtype=complex)  # 常规波束形成器的输出功率
    for index in range(L_thetaArray):
        tmp = np.exp(-2j * np.pi * freq0 * delta / 1500 * np.arange(0, M, 1) * np.sin(thetaRad[index]))
        weight[index, :] = tmp
        P[index] = tmp.conjugate() @ R_hat @ tmp

    PdB = 20 * np.log10(abs(P))
    return thetaDeg, PdB


if __name__ == '__main__':
    print("================== ConventionalBeamForming ==================")

    print("================== プログラム終了 ==================")
