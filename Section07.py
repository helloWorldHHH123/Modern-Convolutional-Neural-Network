# -*- coding:utf-8 -*-
'''
作者：cy
日期：2025年11月06日
7.7 稠密连接网络（DenseNet）

ResNet极大地改变了如何参数化深层网络中函数的观点。
稠密连接网络（DenseNet）(Huang et al., 2017)在某种程度上是ResNet的逻辑扩展。

ResNet将f分解为两部分：一个简单的线性项和一个复杂的非线性项。那么再向前拓展一步，
如果我们想将f拓展成超过两部分的信息呢？一种方案便是DenseNet。

ResNet和DenseNet的关键区别在于，DenseNet输出是连接（用图中的[, ]表示）
而不是如ResNet的简单相加。
x → [x, f1(x), f2([x, f1(x)]), f3([x, f1(x), f2([x, f1(x)])]), . . .]
而ResNet: f(x) = x + g(x)
'''

import Section01
import torch
from torch import nn


# 7.7.2 稠密块体
# DenseNet使用了ResNet改良版的“批量规范化、激活和卷积”架构
def conv_block(input_channels, num_channels):
    return nn.Sequential(
        nn.BatchNorm2d(input_channels), nn.ReLU(),
        nn.Conv2d(input_channels, num_channels, kernel_size=3, padding=1))


# 一个稠密块由多个卷积块组成，每个卷积块使用相同数量的输出通道。
# 然而，在前向传播中，我们将每个卷积块的输入和输出在通道维上连结。
class DenseBlock(nn.Module):
    def __init__(self, num_convs, input_channels, num_channels):
        super(DenseBlock, self).__init__()
        layer = []
        for i in range(num_convs):
            layer.append(conv_block(
                num_channels * i + input_channels, num_channels))
        self.net = nn.Sequential(*layer)

    def forward(self, X):
        for blk in self.net:
            Y = blk(X)
            # 连接通道维度上每个块的输入和输出
            X = torch.cat((X, Y), dim=1)
        return X

# 由于每个稠密块都会带来通道数的增加，使用过多则会过于复杂化模型。而过渡层可以用来控制模型复杂度。
# 它通过1 × 1卷积层来减小通道数，并使用步幅为2的平均汇聚层减半高和宽，从而进一步降低模型复杂度。
def transition_block(input_channels, num_channels):
    return nn.Sequential(
        nn.BatchNorm2d(input_channels), nn.ReLU(),
        nn.Conv2d(input_channels, num_channels, kernel_size=1),
        nn.AvgPool2d(kernel_size=2, stride=2))

def main():
    """主函数，包含所有需要执行的代码"""
    print("🚀 Section07.py 的主函数")
    blk = DenseBlock(2, 3, 10)
    X = torch.randn(4, 3, 8, 8)
    Y = blk(X)
    print('Y.shape = ',Y.shape)
    blk = transition_block(23, 10)
    print('blk(Y).shape = ',blk(Y).shape)

    # 构造DenseNet模型。DenseNet首先使用同ResNet一样的单卷积层和最大汇聚层
    b1 = nn.Sequential(
        nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
        nn.BatchNorm2d(64), nn.ReLU(),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

    # 接下来，类似于ResNet使用的4个残差块，DenseNet使用的是4个稠密块。
    # 与ResNet类似，我们可以设置每个稠密块使用多少个卷积层。这里设成4
    # num_channels为当前的通道数
    num_channels, growth_rate = 64, 32
    num_convs_in_dense_blocks = [4, 4, 4, 4]
    blks = []
    for i, num_convs in enumerate(num_convs_in_dense_blocks):
        blks.append(DenseBlock(num_convs, num_channels, growth_rate))
        # 上一个稠密块的输出通道数
        num_channels += num_convs * growth_rate
        # 在稠密块之间添加一个转换层，使通道数量减半
        if i != len(num_convs_in_dense_blocks) - 1:
            blks.append(transition_block(num_channels, num_channels // 2))
            num_channels = num_channels // 2

    # 与ResNet类似，最后接上全局汇聚层和全连接层来输出结果。
    net = nn.Sequential(
        b1, *blks,
        nn.BatchNorm2d(num_channels), nn.ReLU(),
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
        nn.Linear(num_channels, 10))

    lr, num_epochs, batch_size = 0.1, 10, 256
    train_iter, test_iter = Section01.load_data_fashion_mnist(batch_size, resize=96)
    Section01.train_ch6(net, train_iter, test_iter, num_epochs, lr, Section01.try_gpu())

if __name__ == '__main__':
    main()

