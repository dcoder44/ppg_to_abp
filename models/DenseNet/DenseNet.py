import torch
import torch.nn as nn
import torch.nn.functional as F

class DenseLayer1D(nn.Module):
    def __init__(self, in_channels, growth_rate):
        super().__init__()
        self.bn = nn.BatchNorm1d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv1d(in_channels, growth_rate, kernel_size=3, padding=1)

    def forward(self, x):
        out = self.conv(self.relu(self.bn(x)))
        out = torch.cat([x, out], dim=1)  # Concatenate input and output
        return out
    
    
class DenseBlock1D(nn.Module):
    def __init__(self, num_layers, in_channels, growth_rate):
        super().__init__()
        layers = []
        channels = in_channels
        for _ in range(num_layers):
            layers.append(DenseLayer1D(channels, growth_rate))
            channels += growth_rate
        self.block = nn.Sequential(*layers)
        self.out_channels = channels  # Keep track of output channels

    def forward(self, x):
        return self.block(x)
    

class TransitionLayer1D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.layer = nn.Sequential(
            nn.BatchNorm1d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels, out_channels, kernel_size=1),
            nn.AvgPool1d(kernel_size=2)
        )

    def forward(self, x):
        return self.layer(x)
    

class DenseNet(nn.Module):
    # def __init__(self, input_channels=1, growth_rate=32, num_blocks=3, layers_per_block=4):
    def __init__(self, config):
        super().__init__()
        
        input_channels = config.get("input_channels", 1)
        growth_rate = config.get("growth_rate", 32)
        num_blocks = config.get("num_blocks", 3)
        layers_per_block = config.get("layers_per_block", 4)
        seq_len = config.get("seq_len", 1000)
        
        self.growth_rate = growth_rate

        # Initial convolution
        self.init_conv = nn.Conv1d(input_channels, 64, kernel_size=7, padding=3)

        channels = 64
        self.blocks = nn.ModuleList()
        self.transitions = nn.ModuleList()

        for i in range(num_blocks):
            block = DenseBlock1D(layers_per_block, channels, growth_rate)
            self.blocks.append(block)
            channels = block.out_channels

            # Add transition layer unless it's the last block
            if i != num_blocks - 1:
                out_channels = channels // 2
                self.transitions.append(TransitionLayer1D(channels, out_channels))
                channels = out_channels

        # Final conv to reduce to 1 output channel
        self.final_conv = nn.Sequential(
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(channels, 1, kernel_size=1)
        )

        self.upsample = nn.Upsample(size=seq_len, mode='linear', align_corners=True)

    def forward(self, x, features):
        x = x.unsqueeze(1)
        out = self.init_conv(x)
        for i, block in enumerate(self.blocks):
            out = block(out)
            if i < len(self.transitions):
                out = self.transitions[i](out)

        out = self.final_conv(out)
        out = self.upsample(out)
        out = out.squeeze(1)
        return out