import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, bias):
        super(ConvLSTMCell, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        self.bias = bias

        self.conv = nn.Conv2d(
            in_channels=self.input_dim + self.hidden_dim,
            out_channels=4 * self.hidden_dim,
            kernel_size=self.kernel_size,
            padding=self.padding,
            bias=self.bias,
        )

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state
        combined = torch.cat([input_tensor, h_cur], dim=1)
        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)

        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (
            torch.zeros(
                batch_size,
                self.hidden_dim,
                height,
                width,
                device=self.conv.weight.device,
            ),
            torch.zeros(
                batch_size,
                self.hidden_dim,
                height,
                width,
                device=self.conv.weight.device,
            ),
        )


class HybridConvLSTMUNet(nn.Module):
    def __init__(self, in_channels, num_classes=1):
        super(HybridConvLSTMUNet, self).__init__()

        # --- Encoder (Spatial Features) ---
        # We process each frame independently first, then fuse time
        self.enc1 = self._conv_block(in_channels, 32)
        self.enc2 = self._conv_block(32, 64)
        self.pool = nn.MaxPool2d(2, 2)

        # --- Temporal Bottleneck (ConvLSTM) ---
        # The LSTM happens at a lower resolution (W/2, H/2) to save memory
        self.conv_lstm = ConvLSTMCell(
            input_dim=64, hidden_dim=64, kernel_size=3, bias=True
        )

        # --- Decoder (Spatial Reconstruction) ---
        self.up2 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec2 = self._conv_block(64, 32)  # 64 because 32 (up) + 32 (skip from enc1)

        self.final_conv = nn.Conv2d(32, num_classes, kernel_size=1)

        # --- Regularization ---
        self.dropout = nn.Dropout2d(
            p=0.3
        )  # Drop 30% of features to prevent overfitting

    def _conv_block(self, in_c, out_c):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        # x shape: (Batch, Seq, Chan, H, W)
        b, seq, c, h, w = x.size()

        # Initialize LSTM states
        # Note: We pool before LSTM, so height/width is halved
        h_t, c_t = self.conv_lstm.init_hidden(b, (h // 2, w // 2))

        # Pass each time step through Encoder -> ConvLSTM
        # We only care about the LAST hidden state for prediction
        # (Many-to-One architecture)

        skip_connection = None

        for t in range(seq):
            xt = x[:, t, :, :, :]  # (B, C, H, W)

            # Spatial Encode
            e1 = self.enc1(xt)  # (B, 32, H, W)
            e1_pool = self.pool(e1)  # (B, 32, H/2, W/2)
            e2 = self.enc2(e1_pool)  # (B, 64, H/2, W/2)

            # Capture skip connection from the LAST frame
            if t == seq - 1:
                skip_connection = e1

            # Temporal Update
            h_t, c_t = self.conv_lstm(e2, (h_t, c_t))

        # h_t is now the spatio-temporal embedding of the sequence
        # Shape: (B, 64, H/2, W/2)

        # Apply Dropout to the bottleneck!
        h_t = self.dropout(h_t)

        # --- Decoder ---
        d2 = self.up2(h_t)  # (B, 32, H, W)

        # Concatenate with skip connection from the last input frame
        d2 = torch.cat([d2, skip_connection], dim=1)  # (B, 64, H, W)
        d2 = self.dec2(d2)  # (B, 32, H, W)

        out = self.final_conv(d2)  # (B, 1, H, W)
        return out
