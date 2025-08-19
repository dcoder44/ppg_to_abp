import torch
import torch.nn as nn

class CNNLSTM(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        input_channels=config.get("input_channels", 1)
        conv_channels=config.get("conv_channels", [32, 64, 128])
        kernel_size=config.get("kernel_size", 7)
        pool_size=config.get("pool_size", 2)
        dropout_cnn=config.get("dropout_cnn", 0.3)
        lstm_hidden_size=config.get("lstm_hidden_size", 128)
        lstm_layers=config.get("lstm_layers", 2)
        dropout_lstm=config.get("dropout_lstm", 0.3)
        seq_len= config.get("seq_len", 1000)

        # === CNN Layers ===
        cnn_layers = []
        in_channels = input_channels
        for out_channels in conv_channels:
            cnn_layers += [
                nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size//2),
                nn.ReLU(),
                nn.MaxPool1d(pool_size),
                nn.Dropout(dropout_cnn)
            ]
            in_channels = out_channels

        self.cnn = nn.Sequential(*cnn_layers)

        # Calculate reduced sequence length after pooling
        downsample_factor = pool_size ** len(conv_channels)
        self.reduced_seq_len = seq_len // downsample_factor
        self.cnn_output_size = conv_channels[-1]  # final channel count

        # === LSTM Layers ===
        self.lstm = nn.LSTM(
            input_size=self.cnn_output_size,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_layers,
            dropout=dropout_lstm,
            batch_first=True
        )

        # === Final Mapping to ABP ===
        self.linear = nn.Linear(lstm_hidden_size, 1)
        self.upsample = nn.Upsample(size=seq_len, mode='linear', align_corners=True)

    def forward(self, x, features):
        x = x.unsqueeze(1)
        # x shape: (batch_size, 1, 1024)
        out = self.cnn(x)  # (batch_size, C, L)

        # Prepare for LSTM: (batch_size, seq_len, features)
        out = out.permute(0, 2, 1)  # (batch_size, reduced_seq_len, C)

        out, _ = self.lstm(out)  # LSTM output: (batch_size, reduced_seq_len, hidden_size)

        out = self.linear(out)  # (batch_size, reduced_seq_len, 1)
        out = out.permute(0, 2, 1)  # (batch_size, 1, reduced_seq_len)

        # Upsample back to original length
        out = self.upsample(out)  # (batch_size, 1, 1024)
        out = out.squeeze(1)
        return out