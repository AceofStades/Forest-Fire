import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================================================================
#  Shared building blocks
# ============================================================================


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, dropout=0.0):
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))
        self.double_conv = nn.Sequential(*layers)

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with MaxPool then DoubleConv"""

    def __init__(self, in_channels, out_channels, dropout=0.0):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2), DoubleConv(in_channels, out_channels, dropout=dropout)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then DoubleConv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(
            in_channels, in_channels // 2, kernel_size=2, stride=2
        )
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        # Handle padding if dimensions are not perfectly divisible
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


# ============================================================================
#  1. UNet
# ============================================================================


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes=1):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)

        # Bottleneck with Dropout to prevent memorization
        self.down4 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(512, 1024, dropout=0.5))

        self.up1 = Up(1024, 512)
        self.drop1 = nn.Dropout2d(0.3)
        self.up2 = Up(512, 256)
        self.drop2 = nn.Dropout2d(0.3)
        self.up3 = Up(256, 128)
        self.up4 = Up(128, 64)
        self.outc = nn.Conv2d(64, n_classes, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.drop1(x)
        x = self.up2(x, x3)
        x = self.drop2(x)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.outc(x)


# ============================================================================
#  2. ConvLSTM Fire Net  (temporal, input: B,T,C,H,W → output: B,1,H,W logits)
# ============================================================================


class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size=3):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.conv = nn.Conv2d(
            input_dim + hidden_dim,
            4 * hidden_dim,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            bias=True,
        )

    def forward(self, x, state):
        h, c = state
        combined = torch.cat([x, h], dim=1)
        gates = self.conv(combined)
        i, f, o, g = torch.split(gates, self.hidden_dim, dim=1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)
        c_next = f * c + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next

    def init_hidden(self, batch, h, w):
        device = self.conv.weight.device
        return (
            torch.zeros(batch, self.hidden_dim, h, w, device=device),
            torch.zeros(batch, self.hidden_dim, h, w, device=device),
        )


class ConvLSTMFireNet(nn.Module):
    """Stacked ConvLSTM → 1×1 conv → logits.  Input (B,T,C,H,W)."""

    def __init__(self, in_channels, n_classes=1, hidden_dims=None, dropout=0.3):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [64, 64, 64]
        self.hidden_dims = hidden_dims

        cells = []
        for idx, hd in enumerate(hidden_dims):
            inp = in_channels if idx == 0 else hidden_dims[idx - 1]
            cells.append(ConvLSTMCell(inp, hd))
        self.cells = nn.ModuleList(cells)
        self.dropout = nn.Dropout2d(dropout)
        self.head = nn.Conv2d(hidden_dims[-1], n_classes, kernel_size=1)

    def forward(self, x):
        b, t, c, h, w = x.size()
        # init hidden per layer
        states = [cell.init_hidden(b, h, w) for cell in self.cells]
        for step in range(t):
            inp = x[:, step]
            for i, cell in enumerate(self.cells):
                states[i] = cell(inp, states[i])
                inp = states[i][0]
        out = self.dropout(states[-1][0])
        return self.head(out)
