import numpy
import pickle
import torch
import itertools
import time
import math
from torch.utils.data import Dataset, DataLoader


class HandGestureNet(torch.nn.Module):
    def __init__(self, n_channels=66, n_classes=14, dropout_probability=0.2):

        super(HandGestureNet, self).__init__()

        self.n_channels = n_channels
        self.n_classes = n_classes
        self.dropout_probability = dropout_probability

        # Layers ----------------------------------------------
        self.all_conv_high = torch.nn.ModuleList([torch.nn.Sequential(
            torch.nn.Conv1d(in_channels=1, out_channels=8, kernel_size=7, padding=3),
            torch.nn.ReLU(),
            torch.nn.AvgPool1d(2),

            torch.nn.Conv1d(in_channels=8, out_channels=4, kernel_size=7, padding=3),
            torch.nn.ReLU(),
            torch.nn.AvgPool1d(2),

            torch.nn.Conv1d(in_channels=4, out_channels=4, kernel_size=7, padding=3),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=self.dropout_probability),
            torch.nn.AvgPool1d(2)
        ) for joint in range(n_channels)])

        self.all_conv_low = torch.nn.ModuleList([torch.nn.Sequential(
            torch.nn.Conv1d(in_channels=1, out_channels=8, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.AvgPool1d(2),

            torch.nn.Conv1d(in_channels=8, out_channels=4, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.AvgPool1d(2),

            torch.nn.Conv1d(in_channels=4, out_channels=4, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=self.dropout_probability),
            torch.nn.AvgPool1d(2)
        ) for joint in range(n_channels)])

        self.all_residual = torch.nn.ModuleList([torch.nn.Sequential(
            torch.nn.AvgPool1d(2),
            torch.nn.AvgPool1d(2),
            torch.nn.AvgPool1d(2)
        ) for joint in range(n_channels)])

        self.fc = torch.nn.Sequential(
            torch.nn.Linear(in_features=9 * n_channels * 12, out_features=1936),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=1936, out_features=n_classes)
        )

        # 初始化 --------------------------------------
        for module in itertools.chain(self.all_conv_high, self.all_conv_low, self.all_residual):
            for layer in module:
                if layer.__class__.__name__ == "Conv1d":
                    torch.nn.init.xavier_uniform_(layer.weight, gain=torch.nn.init.calculate_gain('relu'))
                    torch.nn.init.constant_(layer.bias, 0.1)

        for layer in self.fc:
            if layer.__class__.__name__ == "Linear":
                torch.nn.init.xavier_uniform_(layer.weight, gain=torch.nn.init.calculate_gain('relu'))
                torch.nn.init.constant_(layer.bias, 0.1)

    def forward(self, input):
        # 单独处理每个通道
        all_features = []

        for channel in range(0, self.n_channels):
            input_channel = input[:, :, channel]

            # 为时间卷积添加虚拟（空间）维度
            # Conv1D 格式 : (batch_size, n_feature_maps, duration)
            input_channel = input_channel.unsqueeze(1)

            high = self.all_conv_high[channel](input_channel)
            low = self.all_conv_low[channel](input_channel)
            ap_residual = self.all_residual[channel](input_channel)

            # 时间卷积沿特征映射轴串联
            output_channel = torch.cat([
                high,
                low,
                ap_residual
            ], dim=1)
            all_features.append(output_channel)

        # 沿特征映射轴串联
        all_features = torch.cat(all_features, dim=1)

        # 展平
        all_features = all_features.view(-1,
                                         9 * self.n_channels * 12)

        # 全连接层
        output = self.fc(all_features)

        return output

