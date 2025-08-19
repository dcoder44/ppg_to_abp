import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=7, stride=1, downsample=False):
        super().__init__()
        padding = kernel_size // 2
        self.downsample = downsample
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, 1, padding)
        self.bn2 = nn.BatchNorm1d(out_channels)

        if downsample or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1, stride),
                nn.BatchNorm1d(out_channels)
            )
        else:
            self.skip = nn.Identity()

    def forward(self, x):
        identity = self.skip(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        out = self.relu(out)
        return out
    
    
class ResNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        input_channels = config.get('input_channels', 1)
        output_channels = config.get('output_channels', 1)
        base_channels = config.get('base_channels', 64)
        seq_len = config.get('seq_len', 1000)
        
        self.initial = nn.Sequential(
            nn.Conv1d(input_channels, base_channels, kernel_size=7, padding=3),
            nn.BatchNorm1d(base_channels),
            nn.ReLU(inplace=True)
        )

        self.layer1 = ResidualBlock1D(base_channels, base_channels)
        self.layer2 = ResidualBlock1D(base_channels, base_channels * 2, downsample=True)
        self.layer3 = ResidualBlock1D(base_channels * 2, base_channels * 4, downsample=True)
        self.layer4 = ResidualBlock1D(base_channels * 4, base_channels * 2)
        self.layer5 = ResidualBlock1D(base_channels * 2, base_channels)

        self.output_layer = nn.Conv1d(base_channels, output_channels, kernel_size=1)

        self.upsample = nn.Upsample(size=seq_len, mode='linear', align_corners=True)

    def forward(self, x, features):
        x = x.unsqueeze(1)
        out = self.initial(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.output_layer(out)
        out = self.upsample(out)
        out = out.squeeze(1)
        return out