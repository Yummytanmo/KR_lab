import torch
import torch.nn as nn
import torch.nn.functional as F


class ModelA(nn.Module):
    def __init__(self):
        super(ModelA, self).__init__()
        self.conv = nn.ModuleList([
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.Conv2d(16, 32, kernel_size=3, padding=1)
        ])

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc = nn.ModuleList([
            nn.Linear(32 * 7 * 7, 128),
            nn.Linear(128, 10)
        ])

    def forward(self, x):
        for layer in self.conv:
            x = F.relu(layer(x))
            x = self.pool(x)
        x = x.view(x.size(0), -1)       
        x = F.relu(self.fc[0](x))
        x = self.fc[1](x)
        return x


class ModelB(nn.Module):
    def __init__(self):
        super(ModelB, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.25),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.25)
        )
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x