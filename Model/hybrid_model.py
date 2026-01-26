import torch
import torch.nn as nn
from convlstm_model import ConvLSTMCell


class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class HybridConvLSTMUNet(nn.Module):
    def __init__(self, in_channels, out_channels=1):
        super().__init__()
        self.enc1 = ConvBlock(in_channels, 32)
        self.enc2 = ConvBlock(32, 64)
        self.pool = nn.MaxPool2d(2)

        # Bottleneck ConvLSTM
        self.bottleneck_lstm = ConvLSTMCell(64, 64, (3, 3), True)

        self.up = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec = ConvBlock(64, 32)  # 32 from up + 32 from skip
        self.final = nn.Conv2d(32, out_channels, 1)

    def forward(self, x):
        b, t, c, h, w = x.size()
        h_s, c_s = self.bottleneck_lstm.init_hidden(b, (h // 2, w // 2))

        last_skip = None
        for step in range(t):
            xt = x[:, step]
            s1 = self.enc1(xt)
            s2 = self.enc2(self.pool(s1))
            h_s, c_s = self.bottleneck_lstm(s2, (h_s, c_s))
            if step == t - 1:
                last_skip = s1

        out = self.up(h_s)
        out = torch.cat([out, last_skip], dim=1)
        return self.final(self.dec(out))
