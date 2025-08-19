import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=9, downsample=False):
        super().__init__()
        stride = 2 if downsample else 1
        padding = kernel_size // 2

        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, stride=1, padding=padding)
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.downsample = None
        if downsample or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        identity = x
        if self.downsample:
            identity = self.downsample(x)

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        return self.relu(out)
    
    
class ResNet1D_50(nn.Module):
    def __init__(self, input_channels=1, filters=[32, 64, 128, 256], kernel_size=9):
        super().__init__()

        self.init = nn.Sequential(
            nn.Conv1d(input_channels, filters[0], kernel_size=kernel_size, padding=kernel_size // 2),
            nn.BatchNorm1d(filters[0]),
            nn.ReLU(inplace=True)
        )
        
        self.layer1 = self._make_layer(filters[0], filters[0], 2, kernel_size, downsample=False)
        self.layer2 = self._make_layer(filters[0], filters[1], 3, kernel_size, downsample=True)
        self.layer3 = self._make_layer(filters[1], filters[2], 3, kernel_size, downsample=True)
        self.layer4 = self._make_layer(filters[2], filters[3], 2, kernel_size, downsample=True)

        self.global_pool = nn.AdaptiveAvgPool1d(1)

    def _make_layer(self, in_channels, out_channels, blocks, kernel_size, downsample):
        layers = [ResidualBlock1D(in_channels, out_channels, kernel_size, downsample)]
        for _ in range(1, blocks):
            layers.append(ResidualBlock1D(out_channels, out_channels, kernel_size))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.init(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.global_pool(out)  # (batch_size, channels, 1)
        return out.squeeze(-1)       # (batch_size, channels)
    
    
class LSTMBranch(nn.Module):
    def __init__(self, input_size=1, hidden_size=128, num_layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers,
                            batch_first=True, dropout=dropout, bidirectional=False)

        self.output_size = hidden_size

    def forward(self, x):
        x = x.squeeze(1).unsqueeze(-1)  # (B, 1000, 1)
        out, _ = self.lstm(x)           # (B, 1000, H)
        out = out[:, -1, :]             # Last time step
        return out                      # (B, H)
    
    
class ResLSTM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.resnet = ResNet1D_50()
        self.lstm = LSTMBranch()
        
        seq_len = config.get('seq_len', 1000)

        concat_dim = 256 + 128  # ResNet output + LSTM output

        self.fc = nn.Sequential(
            nn.Linear(concat_dim, 256),
            nn.ReLU(),
            nn.Linear(256, seq_len)  # Output ABP waveform
        )

    def forward(self, x, features):
        x = x.unsqueeze(1)
        res_out = self.resnet(x)  # (B, 128)
        lstm_out = self.lstm(x)   # (B, 128)

        combined = torch.cat([res_out, lstm_out], dim=1)  # (B, 256)
        abp_out = self.fc(combined)                       # (B, 1000)
        return abp_out                                    # (B, 1, 1000)