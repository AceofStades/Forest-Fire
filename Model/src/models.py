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


class AttentionGate(nn.Module):
    """Attention gate for skip connections (Attention U-Net, Oktay et al.)"""

    def __init__(self, gate_ch, skip_ch, inter_ch):
        super().__init__()
        self.W_gate = nn.Sequential(
            nn.Conv2d(gate_ch, inter_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(inter_ch),
        )
        self.W_skip = nn.Sequential(
            nn.Conv2d(skip_ch, inter_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(inter_ch),
        )
        self.psi = nn.Sequential(
            nn.Conv2d(inter_ch, 1, kernel_size=1, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, gate, skip):
        g = self.W_gate(gate)
        s = self.W_skip(skip)
        # Align spatial dims (gate may be smaller)
        g = F.interpolate(g, size=s.shape[2:], mode="bilinear", align_corners=False)
        att = self.psi(self.relu(g + s))
        return skip * att


class AttentionUp(nn.Module):
    """Up-sampling block with attention gate on the skip connection."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(
            in_channels, in_channels // 2, kernel_size=2, stride=2
        )
        self.att = AttentionGate(
            gate_ch=in_channels // 2, skip_ch=in_channels // 2, inter_ch=in_channels // 4
        )
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])

        x2 = self.att(gate=x1, skip=x2)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


# ============================================================================
#  1. Original UNet (kept for backward compatibility with evaluate.py)
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
        self.down4 = nn.Sequential(
            nn.MaxPool2d(2), DoubleConv(512, 1024), nn.Dropout(0.5)
        )

        self.up1 = Up(1024, 512)
        self.up2 = Up(512, 256)
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
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.outc(x)


# ============================================================================
#  2. Attention U-Net  (single-frame, outputs logits)
# ============================================================================

class AttentionUNet(nn.Module):
    def __init__(self, n_channels, n_classes=1, base_filters=64, dropout=0.3):
        super().__init__()
        f = base_filters
        self.inc = DoubleConv(n_channels, f)
        self.down1 = Down(f, f * 2, dropout=dropout)
        self.down2 = Down(f * 2, f * 4, dropout=dropout)
        self.down3 = Down(f * 4, f * 8, dropout=dropout)
        self.bottleneck = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(f * 8, f * 16, dropout=dropout),
        )
        self.up1 = AttentionUp(f * 16, f * 8)
        self.up2 = AttentionUp(f * 8, f * 4)
        self.up3 = AttentionUp(f * 4, f * 2)
        self.up4 = AttentionUp(f * 2, f)
        self.outc = nn.Conv2d(f, n_classes, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.bottleneck(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.outc(x)


# ============================================================================
#  3. ConvLSTM Fire Net  (temporal, input: B,T,C,H,W → output: B,1,H,W logits)
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


# ============================================================================
#  4. Hybrid ConvLSTM-UNet  (temporal encoder + spatial decoder, outputs logits)
# ============================================================================

class HybridFireNet(nn.Module):
    """Per-frame UNet encoder → ConvLSTM bottleneck → decoder with proper attention skips."""

    def __init__(self, in_channels, n_classes=1, base_filters=32, dropout=0.3):
        super().__init__()
        f = base_filters

        # Shared spatial encoder (applied per frame)
        self.enc1 = DoubleConv(in_channels, f)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = DoubleConv(f, f * 2, dropout=dropout)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = DoubleConv(f * 2, f * 4, dropout=dropout)
        self.pool3 = nn.MaxPool2d(2)

        # Temporal bottleneck at 1/8 resolution
        self.conv_lstm = ConvLSTMCell(f * 4, f * 4)
        self.bottleneck_drop = nn.Dropout2d(dropout)

        # Decoder level 3: H/8 → H/4  (concat with enc3 skip)
        self.up3_t   = nn.ConvTranspose2d(f * 4, f * 4, kernel_size=2, stride=2)
        self.att3    = AttentionGate(gate_ch=f * 4, skip_ch=f * 4, inter_ch=f * 2)
        self.conv3   = DoubleConv(f * 8, f * 2)

        # Decoder level 2: H/4 → H/2  (concat with enc2 skip)
        self.up2_t   = nn.ConvTranspose2d(f * 2, f * 2, kernel_size=2, stride=2)
        self.att2    = AttentionGate(gate_ch=f * 2, skip_ch=f * 2, inter_ch=f)
        self.conv2   = DoubleConv(f * 4, f)

        # Decoder level 1: H/2 → H    (concat with enc1 skip)
        self.up1_t   = nn.ConvTranspose2d(f, f, kernel_size=2, stride=2)
        self.att1    = AttentionGate(gate_ch=f, skip_ch=f, inter_ch=f // 2)
        self.conv1   = DoubleConv(f * 2, f)

        self.outc = nn.Conv2d(f, n_classes, kernel_size=1)

    def _align(self, x, ref):
        """Bilinear resize x to match ref spatial dims if needed."""
        if x.shape[2:] != ref.shape[2:]:
            x = F.interpolate(x, size=ref.shape[2:], mode="bilinear", align_corners=False)
        return x

    def forward(self, x):
        # x: (B, T, C, H, W)
        b, t, c, h, w = x.size()

        state = self.conv_lstm.init_hidden(b, h // 8, w // 8)
        skip1 = skip2 = skip3 = None

        for step in range(t):
            frame = x[:, step]
            e1 = self.enc1(frame)               # (B, f,   H,   W)
            e2 = self.enc2(self.pool1(e1))      # (B, 2f,  H/2, W/2)
            e3 = self.enc3(self.pool2(e2))      # (B, 4f,  H/4, W/4)
            state = self.conv_lstm(self.pool3(e3), state)
            if step == t - 1:
                skip1, skip2, skip3 = e1, e2, e3

        lstm_out = self.bottleneck_drop(state[0])   # (B, 4f, H/8, W/8)

        # Level 3
        d3 = self._align(self.up3_t(lstm_out), skip3)   # (B, 4f, H/4, W/4)
        s3 = self.att3(gate=d3, skip=skip3)
        d3 = self.conv3(torch.cat([d3, s3], dim=1))     # (B, 2f, H/4, W/4)

        # Level 2
        d2 = self._align(self.up2_t(d3), skip2)         # (B, 2f, H/2, W/2)
        s2 = self.att2(gate=d2, skip=skip2)
        d2 = self.conv2(torch.cat([d2, s2], dim=1))     # (B, f,  H/2, W/2)

        # Level 1
        d1 = self._align(self.up1_t(d2), skip1)         # (B, f,  H,   W)
        s1 = self.att1(gate=d1, skip=skip1)
        d1 = self.conv1(torch.cat([d1, s1], dim=1))     # (B, f,  H,   W)

        return self.outc(d1)
