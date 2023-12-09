# -*- coding: utf-8 -*-
"""
    @File: modelTest.py
    @Author: Chaos
    @Date: 2023/6/20
    @Description: 
"""
import pickle
import numpy as np
import torch.cuda
from pathLoader import getpath
# 导入网络模型对象
from CNNmodel.NetModel import NetM32


if __name__ == '__main__':
    print("================== modelValidation ==================")
    testSetPath = getpath("testSetPath")
    modelPath = getpath("netModelPath")
    
    print("测试集路径：", testSetPath)
    print("网络模型路径：", modelPath)

    with open(testSetPath, 'rb') as f:
        testSet = pickle.load(f)
    # testLoader = torch.utils.data.DataLoader(testSet, batch_size=1, shuffle=True)

    model = NetM32()
    model.load_state_dict(torch.load(modelPath))
    if torch.cuda.is_available():
        model.cuda()

    correct = 0
    total = 0
    with torch.no_grad():
        for (inputs, bearingLabels) in testSet:
            if torch.cuda.is_available():
                inputs, bearingLabels = inputs.cuda(), bearingLabels.cuda()
            outputBearing = model(inputs)
            _, predicted = torch.max(outputBearing.data, dim=1)  # 输出预测的向量中最大值的位置
            reality = np.argmax(np.array(bearingLabels.cpu()))
            if predicted.item() == reality.item():
                correct += 1
            total += 1
    accuracyRate = 100 * correct / total
    print('Accuracy of the network on the test set: %d %%' % (accuracyRate))
