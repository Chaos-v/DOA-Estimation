# -*- coding: utf-8 -*-
"""
    @File: test.py.py
    @Author: Chaos
    @Created: 2023/12/11
    @Description: 
"""
import os
from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
# from CNNmodel import NetModel


if __name__ == '__main__':

    config = {
        "font.family": "sans-serif",
        "font.size": 14,
        "savefig.dpi": 240
    }
    rcParams.update(config)

    print("================== test.py ==================")

    files = ".\\CNNmodel\\model_M32"
    matFile = "M32_SNRG_mParam.mat"
    matPath = os.path.join(files, matFile)

    matLossData = loadmat(matPath)

    L = matLossData['TrainingLoss'].size
    x = np.arange(L).reshape((L, 1))

    figLoss, axLoss = plt.subplots(figsize=(6.67, 5), layout='constrained')
    axLoss.plot(matLossData['TrainingLoss'].T, label="Training Loss")
    axLoss.plot(matLossData['ValidationLoss'].T, label="Validation Loss")
    axLoss.set_xlabel("Epoch")
    axLoss.set_ylabel("Loss")
    axLoss.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    axLoss.legend()
    axLoss.grid(True)
    # figLoss.show()
    figLoss.savefig("loss.png")

    figAcc, axAcc = plt.subplots(figsize=(6.67, 5), layout='constrained')
    axAcc.plot(x, matLossData['TrainingAccuracy '].T, label="Training Acc")
    axAcc.plot(x, matLossData['ValidationAccuracy'].T, label="Validation Acc")
    axAcc.set_ylim(0.9, 1)
    axAcc.set_xlabel("Epoch")
    axAcc.set_ylabel("Acc")
    axAcc.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    axAcc.legend(loc="lower right")
    axAcc.grid(True)
    figAcc.savefig("acc.png")
    # plt.show()

    correctRate = float(matLossData['correctRate'][0][0]) * 100
    print("正确率：%f%%" % correctRate)
    print("================== プログラム終了 ==================")
    # axAcc.plot(x[:16], matLossData['ValidationAccuracy'][0][:16].reshape(1, 16).T, label="Validation Acc")
