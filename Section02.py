# -*- coding:utf-8 -*-
'''
作者：cy
日期：2025年11月05日
7.2 使用块的网络（VGG）

经典卷积神经网络的基本组成部分是下面的这个序列：
1. 带填充以保持分辨率的卷积层；
2. 非线性激活函数，如ReLU；
3. 汇聚层，如最大汇聚层。
'''
import Section01
import torch
from torch import nn


# 7.2.1 VGG块
def vgg_block(num_convs, in_channels, out_channels):
    layers = []
    for _ in range(num_convs):
        layers.append(nn.Conv2d(in_channels, out_channels,
                                kernel_size=3, padding=1))
        layers.append(nn.ReLU())
        in_channels = out_channels
    layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
    return nn.Sequential(*layers)


# 7.2.2 VGG网络
# VGG神经网络连接几个VGG块（在vgg_block函数中定义）。
# 原始VGG网络有5个卷积块，其中前两个块各有一个卷积层，后三个块各包含两个卷积层。第一个模块有64个
# 输出通道，每个后续模块将输出通道数量翻倍，直到该数字达到512。由于该网络使用8个卷积层和3个全连接
# 层，因此它通常被称为VGG‐11。
def vgg(conv_arch):
    conv_blks = []
    in_channels = 1
    # 卷积层部分
    for (num_convs, out_channels) in conv_arch:
        conv_blks.append(vgg_block(num_convs, in_channels, out_channels))
        in_channels = out_channels

    return nn.Sequential(
        *conv_blks, nn.Flatten(),
        # 全连接层部分
        nn.Linear(out_channels * 7 * 7, 4096), nn.ReLU(), nn.Dropout(0.5),
        nn.Linear(4096, 4096), nn.ReLU(), nn.Dropout(0.5),
        nn.Linear(4096, 10))


def main():
    """主函数，包含所有需要执行的代码"""
    print("🚀 Section02.py 的主函数")
    # 7.2.2 VGG网络
    conv_arch = ((1, 64), (1, 128), (2, 256), (2, 512), (2, 512))
    net2 = vgg(conv_arch)
    X = torch.randn(size=(1, 1, 224, 224))
    for blk in net2:
        # print('Section02.py文件正在输出...')   # 测试Section01.py文件在不在运行
        X = blk(X)
        print(blk.__class__.__name__, 'output shape:\t', X.shape)
    # 7.2.3 训练模型
    #  由于VGG‐11比AlexNet计算量更大，因此我们构建了一个通道数较少的网络，足够用于训练Fashion‐MNIST数据集。
    ratio = 4
    small_conv_arch = [(pair[0], pair[1] // ratio) for pair in conv_arch]
    net3 = vgg(small_conv_arch)
    lr, num_epochs, batch_size = 0.05, 10, 64
    train_iter, test_iter = Section01.load_data_fashion_mnist(batch_size, resize=224)
    Section01.train_ch7(net3, train_iter, test_iter, num_epochs, lr, Section01.try_gpu())

# 只有直接运行本文件时才执行main()
if __name__ == '__main__':
    main()
