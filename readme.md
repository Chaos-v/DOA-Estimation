# README

@Data: 2023-06-06 

---

## 一、项目说明

新建虚拟环境，本项目预计主要提供两种方位估计算法：

- 常规波束形成 (CBF) 的目标方位估计算法
- 基于卷积神经网络 (CNN) 的目标方位估计算法

截至目前，已完成基于 (CBF) 的目标方位估计算法，能够通过给定一段水平阵列的实采数据，通过滑动时间窗的方法可以给出目标的方位时间 (BTR) 图。初步完成基于 CNN 的滑动时间窗目标方位估计方法，但是该方法在方位估计上具有较大的误差，有待进一步调参训练完善模型。

---

## 二、项目目录结构

```
   ---
    ├ CBF
    ├ CNNmodel
    │   ├ model
    │   ├ NetModel.py
    │   └ mData.bty
    ├ dataraw
    └ dataset
```

- `CBF` 文件夹存放常规波束形成方法，包括一个算法主体的 `py` 文件，一个测试文件；
- `CNNmodel` 文件夹存放卷积神经网络的算法：
  - 文件夹部分包括存放训练模型的子文件夹 `model`
  - 文件部分包含三个 `py` 文件，一个算法主体函数以及辅助函数的文件，一个网络结构的对象库，一个测试文件
- `dataraw` 文件夹存放原始数据；
- `dataset` 文件夹存放数据集文件，内部各子文件夹中分别包括训练集、验证集、测试集。

---

## 三、算法调用说明

### CBF方法

常规波束形成方法位于 `.//CBF//*`，常规波束形成算法位于`ConvBeamForming.py` 文件中。封装思路是部分阵元参数一经设置就不再更改，直接调用 `mainCBF` 函数即可给出结果，所以在使用前需要根据阵列实际信息进行设置。

#### 方法原理

1. 截取阵列采集的一段时间数据并且做 Hilbert 变换；
2. 设置滑动时间窗截取信号调用 CBF 算法进行计算；
3. 生成方位时间图

#### 使用方法

1. 封装在软件前，根据实际阵列信息对方法中的相关进行设置 **(该步骤测试时可忽略)** ；

   - 滑动窗函数 `slideWinView` 的输入变量 `win`，`overlap` 的默认值。分别表示滑动窗长度的快拍数以及相邻两个滑动窗重叠的快拍数。可以直接修改函数中的默认值，也可直接在调用函数时设置默认值。

   - 常规波束形成算法 `CBFCore` 函数的输入变量 `freq0`, `delta` 的默认值。`freq0` 的默认值可直接在 `mainCBF` 函数中修改，修改后直接传递到对应函数，也可直接修改函数中的默认值。变量 `delta` 的值需要单独在 `CBFCore` 函数中进行修改。

2. 直接调用入口函数，按照要求给定参数即可。
    ```python
    from ConvBeamForming import mainCBF

    P_dB, theta_Deg, fig = mainCBF(signal, f0)
    ```

### CNN方法

基于卷积神经网络的滑动时间窗 DOA 估计方法，网络模型基于某论文搭建。`./model` 文件夹中用于存放模型文件，`NetModel.py` 文件存放网络结构类，主要方法在 `DOAEstimation.py` 中。直接调用 `CNN_DOAEstimation` 函数即可给出结果。

#### 使用方法

1. 给定要求的阵列信号数据；
2. 从神经网络模型对象库中导入需要的网络模型类 `(class)`；
3. 更改 `CNN_DOAEstimation` 中网络模型路径参数，使其加载的对象与导入的类对应；
4. 调用 `CNN_DOAEstimation` 方法计算即可。
    
   ```python
    from NetModel import Net
    from DOAEstimation import CNN_DOAEstimation
    
    bearingProbability, thetaDeg, fig = CNN_DOAEstimation(signal)
    ```

