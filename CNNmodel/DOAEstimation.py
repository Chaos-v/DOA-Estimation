# -*- coding: utf-8 -*-
"""
    @ProjectName: CNN_DOA
    @File: DOAEstimation.py
    @Author: Chaos
    @Date: 2023/6/19
    @Description: 基于 CNN 的 DOA 方法，包括算法的函数封装以及辅助函数，主函数：CNN_DOAEstimation
"""
import numpy as np
from scipy.signal import hilbert
# 以下是CNN_DOAEstimation函数需要的包
import torch
import matplotlib.pyplot as plt
from NetModel import NetM20


def CNN_DOAEstimation(sigDict: dict, win=56320, overlap=5120):
    """
    完整的基于 CNN 的DOA算法的封装，用于软件内部调用，使用前需要设置模型路径
    :param sigDict: 包含阵列信号的字典变量，其中阵列信号部分中：阵元 × 采集信号（dim0 × dim1）
    :param win: 表示滑动窗长度的快拍数。长度 = 采样频率 × 窗口时间长度。
    :param overlap: 相邻两个滑动窗重叠的快拍数。长度 = 采样频率 × 重叠部分的时间长度。
    :return:
    """
    # ==================== 模型加载 ====================
    modelPath = ".\\CNNmodel\\model\\M20_SNRG_model.pth"
    model = NetM20()
    model.load_state_dict(torch.load(modelPath))
    if torch.cuda.is_available():
        model.cuda()

    # ==================== 模型计算部分 ====================
    sig = sigPreProcess(sigDict)  # 阵列数据预处理

    vSignal = slideWinView(sig, win, overlap)  # 设置滑动窗截取
    dim0, dim1, dim2 = vSignal.shape

    v = np.zeros((2, dim1, dim1), dtype=np.float32)  # md，还得强制转换一下...
    with torch.no_grad():
        for i, vSig in enumerate(vSignal, 0):
            R_matrix = vSig @ vSig.T.conjugate() / np.size(vSig, 1)
            v[0] = R_matrix.real
            v[1] = R_matrix.imag
            inputs = torch.tensor(v)
            if torch.cuda.is_available():
                inputs = inputs.cuda()
            outputs = model(inputs)
            outputs = np.array(outputs.cpu())
            if i == 0:
                bearingProbability = outputNormalize(outputs)
            else:
                bearingProbability = np.vstack([bearingProbability, outputNormalize(outputs)])

    thetaDeg = np.linspace(-90, 90, 181)
    fig = plt.figure()
    plt.pcolor(thetaDeg, np.linspace(0, dim0 - 1, dim0), bearingProbability)
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.colorbar()
    plt.xlabel('角度（°）')
    plt.ylabel('时间窗')

    return bearingProbability, thetaDeg, fig


def sigPreProcess(sigDict: dict):
    """
    针对传入的实际采集的阵列信号进行预处理，预处理方法根据实际需要添加在该函数中，最终输出预处理后的阵列信号变量。
    该函数可以根据实际需要进行调用。
    :param sigDict: 阵列信号，阵元 × 采集信号（行 × 列）
    :return: arraySig
    """
    # 从传入的字典变量中提取需要信息
    sig = sigDict["signal"]

    m = np.size(sig, 0)  # 输入信号矩阵的行数，也即阵元的数量
    n = np.size(sig, 1)  # 输入信号矩阵的列数

    # Hilbert 变换
    arraySig = np.zeros((m, n), dtype=complex)
    for i in range(m):
        tmp = sig[i, :]
        arraySig[i, :] = hilbert(tmp)

    return arraySig


def slideWinView(sig, win=10, overlap=0):
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


def outputNormalize(v):
    """
    归一化神经网络模型的输出，使其变为概率向量
    :param v: 神经网络的输入向量
    :return: 概率向量
    """
    vRange = np.max(v) - np.min(v)
    vp = (v - np.min(v)) / vRange
    vp[vp < 0.95] = 0  # 可以让预测结果更直观
    return vp


if __name__ == '__main__':
    print("================== Preprocess ==================")
