# -*- coding: utf-8 -*-
"""
    @File: validationAlg.py
    @Author: Chaos
    @Created: 2023/12/14
    @Description: 验证算法的有效性，利用自相关矩阵
"""
import numpy as np
from scipy.io import loadmat
from scipy.signal import hilbert
import matplotlib.pyplot as plt
from signalGenerator import signalGenerate


def sigPreProcess(sigDict, win=51200, overlap=51200):
    # 从传入的字典变量中提取需要信息
    # sig = sigDict["signal"]
    sig = sigDict
    m = np.size(sig, 0)  # 输入信号矩阵的行数，也即阵元的数量
    n = np.size(sig, 1)  # 输入信号矩阵的列数
    # Hilbert 变换
    arraySig = np.zeros((m, n), dtype=complex)
    for i in range(m):
        tmp = sig[i, :]
        arraySig[i, :] = hilbert(tmp)

    shape = (np.size(arraySig, 0), int(win))
    v = np.lib.stride_tricks.sliding_window_view(arraySig, shape)[:, ::overlap, :].squeeze()
    if len(v.shape) == 2:
        vSig = v
    else:
        vSig = v[0]
    R_matrix = vSig @ vSig.T.conjugate() / np.size(vSig, 1)
    R = {'real': R_matrix.real, 'imag': R_matrix.imag}

    return R


if __name__ == '__main__':
    print("================== validationAlg ==================")

    # dataPath = ".\\dataraw\\sdata_M10theta-25snr20.mat"
    # signalDict1 = loadmat(dataPath)
    sig1, _ = signalGenerate(theta0=-25, sampleTime=1, eleNum=10, freq0=500, SNR=20)
    sig2, _ = signalGenerate(theta0=-24, sampleTime=1, eleNum=10, freq0=500, SNR=20)

    R1 = sigPreProcess(sig1)
    R2 = sigPreProcess(sig2)

    fig, ax = plt.subplots(2, 2)
    ax[0, 0].imshow(R1['real'])
    ax[0, 0].set_title('R1 Real')

    ax[0, 1].imshow(R1['imag'])
    ax[0, 1].set_title('R1 Imag')

    ax[1, 0].imshow(R2['real'])
    ax[1, 0].set_title('R2 Real')
    ax[1, 1].imshow(R2['imag'])
    ax[1, 1].set_title('R2 Imag')
    plt.show()
