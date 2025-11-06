"""
作者：cy
时间：2025-11-05
内容：7.1 深度卷积神经网络（AlexNet）
对比：LeNet在小数据集上取得了很好的效果，但是在更大、更真实的数据集上训练卷积神经网络的性能和可行性还有待研究。
AlexNet和LeNet的设计理念非常相似，但也存在显著差异：
1. AlexNet比相对较小的LeNet5要深得多。AlexNet由八层组成：五个卷积层、两个全连接隐藏层和一个
全连接输出层。
2. AlexNet使用ReLU而不是sigmoid作为其激活函数。
3. 为了进一步扩充数据，AlexNet在训练时增加了大量的图像增强数据，如翻转、裁切和变色。这使得模型更健壮，更大的样本量
有效地减少了过拟合。

背景：
2012年，AlexNet横空出世。它首次证明了学习到的特征可以超越手工设计的特征。
Alex Krizhevsky、Ilya Sutskever和Geoff Hinton提出了一种新的卷积神经网络变体AlexNet。
在2012年ImageNet挑战赛中取得了轰动一时的成绩。
在网络的最底层，模型学习到了一些类似于传统滤波器的特征抽取器。

深度卷积神经网络的突破出现在2012年。突破可归因于两个关键因素。
1.缺少的成分：数据
包含许多特征的深度模型需要大量的有标签数据，才能显著优于基于凸优化的传统方法（如线性方法和核方法）。
2009年，ImageNet数据集发布，并发起ImageNet挑战赛：要求研究人员从100万个样本中训练模型，以区分1000个不同
类别的对象。ImageNet数据集由斯坦福教授李飞飞小组的研究人员开发，利用谷歌图像搜索（Google Image
Search）对每一类图像进行预筛选，并利用亚马逊众包（Amazon Mechanical Turk）来标注每张图片的相关
类别。这种规模是前所未有的。这项被称为ImageNet的挑战赛推动了计算机视觉和机器学习研究的发展，挑
战研究人员确定哪些模型能够在更大的数据规模下表现最好。
2.缺少的成分：硬件
当Alex Krizhevsky和Ilya Sutskever实现了可以在GPU硬件上运行的深度卷积神经网络时，
一个重大突破出现了。他们意识到卷积神经网络中的计算瓶颈：卷积和矩阵乘法，都是可以在硬件上并行化的操作。
于是，他们使用两个显存为3GB的NVIDIA GTX580 GPU实现了快速卷积运算。他们的创新cuda‐convnet89
几年来它一直是行业标准，并推动了深度学习热潮。
"""

import torch
from torch import nn
import torchvision
from torchvision import transforms
from torch.utils import data
import time
import numpy as np
import matplotlib.pyplot as plt

net = nn.Sequential(
    # 这里使用一个11*11的更大窗口来捕捉对象。
    # 同时，步幅为4，以减少输出的高度和宽度。
    # 另外，输出通道的数目远大于LeNet
    nn.Conv2d(1, 96, kernel_size=11, stride=4, padding=1), nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2),
    # 减小卷积窗口，使用填充为2来使得输入与输出的高和宽一致，且增大输出通道数
    nn.Conv2d(96, 256, kernel_size=5, padding=2), nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2),
    # 使用三个连续的卷积层和较小的卷积窗口。
    # 除了最后的卷积层，输出通道的数量进一步增加。
    # 在前两个卷积层之后，汇聚层不用于减少输入的高度和宽度
    nn.Conv2d(256, 384, kernel_size=3, padding=1), nn.ReLU(),
    nn.Conv2d(384, 384, kernel_size=3, padding=1), nn.ReLU(),
    nn.Conv2d(384, 256, kernel_size=3, padding=1), nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2),
    nn.Flatten(),
    # 这里，全连接层的输出数量是LeNet中的好几倍。使用dropout层来减轻过拟合
    nn.Linear(6400, 4096), nn.ReLU(),
    nn.Dropout(p=0.5),
    nn.Linear(4096, 4096), nn.ReLU(),
    nn.Dropout(p=0.5),
    # 最后是输出层。由于这里使用Fashion-MNIST，所以用类别数为10，而非论文中的1000
    nn.Linear(4096, 10))


# 7.1.3 读取数据集
# 尽管原文中AlexNet是在ImageNet上进行训练的，但本书在这里使用的是Fashion‐MNIST数据集。
# 因为即使在现代GPU上，训练ImageNet模型，同时使其收敛可能需要数小时或数天的时间。
# 将AlexNet直接应用于Fashion‐MNIST的一个问题是，
# Fashion‐MNIST图像的分辨率（28 × 28像素）低于ImageNet图像。为了解决这个问题，
# 我们将它们增加到224 × 224（通常来讲这不是一个明智的做法，但在这里这样做是为了有效使用AlexNet架构）

# 返回一个批量batch_size的数据（训练+测试）
def load_data_fashion_mnist(batch_size,resize=None):
    # 图像处理顺序很重要：先调整大小，再转换为张量
    # trans 变量本质上就是一个图像预处理的操作序列（流水线或列表）
    trans = [transforms.ToTensor()]  # 1. 创建基础列表
    print("type(trans) = ", type(trans))
    if resize:      # 2. 条件添加resize
        trans.insert(0, transforms.Resize(resize))
    trans = transforms.Compose(trans)     # 3. 组合流水线
    mnist_train = torchvision.datasets.FashionMNIST(root="./data",train=True,transform=trans,download=True)
    mnist_test = torchvision.datasets.FashionMNIST(root="./data",train=False,transform=trans,download=True)
    return data.DataLoader(mnist_train,batch_size,shuffle=True), data.DataLoader(mnist_test,batch_size,shuffle=False)

# 7.1.4 训练AlexNet

class Timer:
    def __init__(self):
        self.times = []
        self.start()
    def start(self):
        """启动计时器"""
        self.tik = time.time()
    def stop(self):
        """停止计时器并将时间记录在列表中"""
        self.times.append(time.time()-self.tik)
        return self.times[-1]
    def sum(self):
        """返回时间总和"""
        return sum(self.times)
    def cumsum(self):
        """返回累计时间"""
        """
        np.array(self.times) - 将列表转换为NumPy数组
        .cumsum() - 计算累积和
        .tolist() - 转换回Python列表
        """
        return np.array(self.times).cumsum().tolist()

def accruacy(y_hat,y):
    # 考虑多分类情况，此时需要取概率最大值
    if len(y_hat.shape)>1 and y_hat.shape[1]>1:
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y.dtype)==y
    return float(cmp.type(y.dtype).sum())

def evaluate_accuracy_gpu(net,test_iter,device = None):
    # 返回测试精度
    if isinstance(net, nn.Module):
        net.eval()   # 设置评估模式，这里会取消反向传播
        if not device:  # 如果device为None时执行，这里的目的是设置cpu还是cuda
            print('device',device)
            # 自动设置网络参数所在的device，即 cuda: 0
            device = next(iter(net.parameters())).device
            # device = net(iter(net.parameters())).device
            print('net device',device)
        # [预测正确个数，总测试样本个数]
        metric = [0.0] * 2
        with torch.no_grad():
            for X,y in test_iter:
                # 首先移动数据到需要的设备上
                if isinstance(X,list):
                    # 将多个张量移至device上
                    X = [x.to(device) for x in X]
                else:
                    X = X.to(device)
                y = y.to(device)
                metric = [a + float(b) for a,b in zip(metric,[accruacy(net(X),y),y.numel()])]
        # 一个批量的数据的测试精度累计完之后
        return metric[0] / metric[1]

def plot_training_curves(train_losses, train_accuracies, test_accuracies, num_epochs):
    """绘制训练曲线"""
    epochs = range(1, num_epochs + 1)

    # 创建子图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # 绘制损失曲线
    ax1.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 绘制准确率曲线
    ax2.plot(epochs, train_accuracies, 'r-', label='Training Accuracy', linewidth=2)
    ax2.plot(epochs, test_accuracies, 'g--', label='Test Accuracy', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Training and Test Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

def train_ch7(net,train_iter,test_iter,num_epochs,lr,device):
    # 初始化权重
    def init_weight(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)
    net.apply(init_weight)
    print('training on', device)
    net.to(device)
    optimizer = torch.optim.SGD(net.parameters(),lr=lr)
    loss = nn.CrossEntropyLoss()
    timer, num_batches = Timer(), len(train_iter)
    # 添加：用于存储历史数据的列表
    train_losses = []
    train_accuracies = []
    test_accuracies = []
    for epoch in range(num_epochs):
        metric = [0.0] * 3
        for i, (X,y) in enumerate(train_iter):
            timer.start()
            optimizer.zero_grad()
            X,y = X.to(device), y.to(device)
            y_hat = net(X)
            l = loss(y_hat,y)
            l.backward()
            optimizer.step()
            with torch.no_grad():
                metric = [a+float(b) for a,b in zip(metric,[l*X.shape[0],accruacy(y_hat,y),X.shape[0]])]
            timer.stop()
            # 当前批次数据的平均训练损失、平均训练精度
            train_loss = metric[0] / metric[2]
            train_acc = metric[1] / metric[2]
        # 一个批量数据结束之后进行测试
        test_acc = evaluate_accuracy_gpu(net, test_iter)
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        test_accuracies.append(test_acc)
        # 打印当前epoch的训练精度和测试精度
        print(f'epoch: {epoch}, train mean accuracy: {train_acc: .3f}, test accuracy: {test_acc: .3f}')
    # 所有epochs结束后，进行画图，打印最后一个epoch的训练损失、训练精度、测试精度、总的时间
    plot_training_curves(train_losses, train_accuracies, test_accuracies, num_epochs)
    print(f'loss {train_loss:.3f}, train acc {train_acc:.3f}, 'f'test acc {test_acc:.3f}')
    # print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec 'f'on {str(device)}')
    print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec 'f'on {device}')


def try_gpu(i=0):
    if torch.cuda.device_count()>=i+1:
        # PyTorch 的设备字符串有严格的格式要求：
        # f'cuda:{i}'，冒号后面没有空格，正确
        # f'cuda: {i}'，冒号后面没有空格，错误
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')

def main():
    """主函数，包含所有需要执行的代码"""
    print("🚀 Section01.py 的主函数")
    X = torch.randn(1, 1, 224, 224)
    for layer in net:
        X = layer(X)
        print(layer.__class__.__name__, 'output shape:\t', X.shape)
    batch_size = 64
    train_iter, test_iter = load_data_fashion_mnist(batch_size, resize=224)
    lr, num_epochs = 0.01, 10
    train_ch7(net, train_iter, test_iter, num_epochs, lr, try_gpu())
# 只有直接运行本文件时才执行main()
if __name__ == '__main__':
    main()
