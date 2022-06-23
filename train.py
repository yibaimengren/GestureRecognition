import numpy
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import time
import math
from model import HandGestureNet
from torch.utils.tensorboard import SummaryWriter
from dataset import GestureDataset,DHG_Dataset
import tool
n_classes = 14
duration = 100
n_channels = 66  #3 * n_joints for 3D pose data
learning_rate = 1e-3

#是否使用数据集，若False则使用随机数据测试
use_dataset = False

if use_dataset:
    #需要下载DHG数据集
    train_dataset = DHG_Dataset()
    test_dataset = DHG_Dataset()
else:
    #模拟数据
    x_train = numpy.random.randn(2000, duration, n_channels)
    y_train = numpy.random.random_integers(n_classes, size=2000)

    x_test = numpy.random.randn(1000, duration, n_channels)
    y_test = numpy.random.random_integers(n_classes, size=1000)

    #获取Dataset和Dataloader
    #将numpy转化为tensor
    x_train, x_test = torch.from_numpy(x_train), torch.from_numpy(x_test)
    y_train, y_test = torch.from_numpy(y_train), torch.from_numpy(y_test)

    # 确保标签值范围在0到class-1
    if y_train.min() > 0:
      y_train = y_train - 1
    if y_test.min() > 0:
      y_test = y_test - 1

    # 确保数据类型正确
    x_train, x_test = x_train.float(), x_test.float()
    y_train, y_test = y_train.long(), y_test.long()

    # datasets
    train_dataset = GestureDataset(x=x_train, y=y_train)
    test_dataset = GestureDataset(x=x_test, y=y_test)

#dataloader
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
test_dataloader = DataLoader(test_dataset,  batch_size=32, shuffle=True, num_workers=0)

model = HandGestureNet() #模型
criterion = torch.nn.CrossEntropyLoss() #损失函数
optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate) #优化器

num_epochs = 20
#开始训练
tool.train(model=model, criterion=criterion, optimizer=optimizer, dataloader=train_dataloader,
      x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test,
      num_epochs=num_epochs)