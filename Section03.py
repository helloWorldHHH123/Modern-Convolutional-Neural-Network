# -*- coding:utf-8 -*-
'''
作者：cy
日期：2025年11月05日
7.3 网络中的网络（NiN）
NiN和AlexNet之间的一个显著区别是NiN完全取消了全连接层。相反，NiN使用一个NiN块，其输出通道数等
于标签类别的数量。NiN设计的一个优点是，它显著减少了模型所需参数的数量。然而，在实践中，这种设计有
时会增加训练模型的时间。

NiN块以一个普通卷积层开始，后面是两个1 × 1的卷积层。
这两个1 × 1卷积层充当带有ReLU激活函数的逐像素全连接层。
第一层的卷积窗口形状通常由用户设置。随后的卷积窗口形状固定为1 × 1。
'''
import Section01
import torch
from torch import nn

# 7.3.1 NiN块
def nin_block(in_channels, out_channels, kernel_size, strides, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, strides, padding),
        nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, kernel_size=1), nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, kernel_size=1), nn.ReLU())

def main():
    """主函数，包含所有需要执行的代码"""
    print("🚀 Section03.py 的主函数")
    # 7.3.2 NiN模型
    net = nn.Sequential(
        nin_block(1, 96, kernel_size=11, strides=4, padding=0),
        nn.MaxPool2d(3, stride=2),
        nin_block(96, 256, kernel_size=5, strides=1, padding=2),
        nn.MaxPool2d(3, stride=2),
        nin_block(256, 384, kernel_size=3, strides=1, padding=1),
        nn.MaxPool2d(3, stride=2),
        nn.Dropout(0.5),
        # 标签类别数是10
        nin_block(384, 10, kernel_size=3, strides=1, padding=1),
        nn.AdaptiveAvgPool2d((1, 1)),
        # 将四维的输出转成二维的输出，其形状为(批量大小,10)
        nn.Flatten())
    X = torch.rand(size=(1, 1, 224, 224))
    for layer in net:
        X = layer(X)
        print(layer.__class__.__name__, 'output shape:\t', X.shape)

    lr, num_epochs, batch_size = 0.1, 10, 128
    train_iter, test_iter = Section01.load_data_fashion_mnist(batch_size, resize=224)
    Section01.train_ch7(net, train_iter, test_iter, num_epochs, lr, Section01.try_gpu())


# 只有直接运行本文件时才执行main()
if __name__ == '__main__':
    main()
