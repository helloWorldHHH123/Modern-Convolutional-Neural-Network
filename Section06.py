# -*- coding:utf-8 -*-
'''
作者：cy
日期：2025年11月06日
7.6 残差网络（ResNet）

残差网络的核心思想是：每个附加层都应该更容易地包含原始函数作为其元素之一。

对于深度神经网络，如果我们能将新添加的层训练成恒等映射（identity function）f(x) = x，
新模型和原模型将同样有效。
'''

import Section01
import torch
from torch import nn
from torch.nn import functional as F

# 7.6.2 残差块
class Residual(nn.Module):
    def __init__(self, input_channels, num_channels,use_1x1conv=False, strides=1):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, num_channels,kernel_size=3, padding=1, stride=strides)
        self.conv2 = nn.Conv2d(num_channels, num_channels,kernel_size=3, padding=1)
        if use_1x1conv:
            # 1*1的卷积层是为了实现将输入直接加在最后的ReLU激活函数前，这是ResNet模型的核心
            # 引入一个额外的1 × 1卷积层来将输入变换成需要的形状后再做相加运算
            self.conv3 = nn.Conv2d(input_channels, num_channels,kernel_size=1, stride=strides)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))   # 卷积1-->规范化-->非线性激活
        Y = self.bn2(self.conv2(Y))     # 卷积2-->规范化
        if self.conv3:
            X = self.conv3(X)
        Y += X
        return F.relu(Y)

# ResNet则使用4个由残差块组成的模块，每个模块使用若干个同样输出通道数的残差块。
def resnet_block(input_channels, num_channels, num_residuals,first_block=False):
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.append(Residual(input_channels, num_channels,use_1x1conv=True, strides=2))
        else:
            blk.append(Residual(num_channels, num_channels))
    return blk


def main():
    """主函数，包含所有需要执行的代码"""
    print("🚀 Section06.py 的主函数")
    blk = Residual(3, 3)
    X = torch.rand(4, 3, 6, 6)
    Y = blk(X)
    print('Y.shape = ',Y.shape)
    blk = Residual(3, 6, use_1x1conv=True, strides=2)
    print('blk(X).shape = ',blk(X).shape)
    # 7.6.3 ResNet模型
    # ResNet的前两层跟之前介绍的GoogLeNet中的一样：在输出通道数为64、步幅为2的7 × 7卷积层后，
    # 接步幅为2的3 × 3的最大汇聚层。不同之处在于ResNet每个卷积层后增加了批量规范化层。
    b1 = nn.Sequential(nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
                       nn.BatchNorm2d(64), nn.ReLU(),
                       nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
    # 接着在ResNet加入所有残差块，这里每个模块使用2个残差块。
    b2 = nn.Sequential(*resnet_block(64, 64, 2, first_block=True))
    b3 = nn.Sequential(*resnet_block(64, 128, 2))
    b4 = nn.Sequential(*resnet_block(128, 256, 2))
    b5 = nn.Sequential(*resnet_block(256, 512, 2))
    # 每个模块有4个卷积层（不包括恒等映射的1 × 1卷积层）。
    # 加上第一个7 × 7卷积层和最后一个全连接层，
    # 共有18层。因此，这种模型通常被称为ResNet‐18。
    net = nn.Sequential(b1, b2, b3, b4, b5,
                        nn.AdaptiveAvgPool2d((1, 1)),
                        nn.Flatten(), nn.Linear(512, 10))
    X = torch.rand(size=(1, 1, 224, 224))
    for layer in net:
        X = layer(X)
        print(layer.__class__.__name__, 'output shape:\t', X.shape)
    lr, num_epochs, batch_size = 0.05, 10, 256
    train_iter, test_iter = Section01.load_data_fashion_mnist(batch_size, resize=96)
    Section01.train_ch7(net, train_iter, test_iter, num_epochs, lr, Section01.try_gpu())

if __name__ == '__main__':
    main()
