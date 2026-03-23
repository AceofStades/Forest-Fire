import os
import sys

import numpy as np
import torch
import torch.nn as nn

# --- 1. Model Definitions ---
# These must match exactly what was used during training/evaluation


def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
    )


class LegacyUNet(nn.Module):
    def __init__(self, in_channels, out_channels=1):
        super().__init__()
        self.dconv_down1 = double_conv(in_channels, 64)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dconv_down2 = double_conv(64, 128)
        self.dconv_down3 = double_conv(128, 256)
        self.dconv_down4 = double_conv(256, 512)
        self.dconv_bottom = double_conv(512, 1024)
        self.upconv_up1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dconv_up1 = double_conv(1024, 512)
        self.upconv_up2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dconv_up2 = double_conv(512, 256)
        self.upconv_up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dconv_up3 = double_conv(256, 128)
        self.upconv_up4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dconv_up4 = double_conv(128, 64)
        self.conv_last = nn.Conv2d(64, out_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        conv1 = self.dconv_down1(x)
        x = self.maxpool(conv1)
        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)
        conv3 = self.dconv_down3(x)
        x = self.maxpool(conv3)
        conv4 = self.dconv_down4(x)
        x = self.maxpool(conv4)
        u = self.dconv_bottom(x)
        u = self.upconv_up1(u)
        u = torch.cat([u, conv4], dim=1)
        u = self.dconv_up1(u)
        u = self.upconv_up2(u)
        u = torch.cat([u, conv3], dim=1)
        u = self.dconv_up2(u)
        u = self.upconv_up3(u)
        u = torch.cat([u, conv2], dim=1)
        u = self.dconv_up3(u)
        u = self.upconv_up4(u)
        u = torch.cat([u, conv1], dim=1)
        u = self.dconv_up4(u)
        out = self.conv_last(u)
        return self.sigmoid(out)


class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, bias):
        super(ConvLSTMCell, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
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


class ConvLSTM(nn.Module):
    def __init__(self, in_channels, out_channels=1, hidden_dims=[32, 32, 32]):
        super(ConvLSTM, self).__init__()
        self.hidden_dims = hidden_dims
        self.num_layers = len(hidden_dims)
        cell_list = []
        cell_list.append(
            ConvLSTMCell(
                input_dim=in_channels,
                hidden_dim=hidden_dims[0],
                kernel_size=(3, 3),
                bias=True,
            )
        )
        for i in range(1, self.num_layers):
            cell_list.append(
                ConvLSTMCell(
                    input_dim=hidden_dims[i - 1],
                    hidden_dim=hidden_dims[i],
                    kernel_size=(3, 3),
                    bias=True,
                )
            )
        self.cell_list = nn.ModuleList(cell_list)
        self.final_conv = nn.Conv2d(hidden_dims[-1], out_channels, kernel_size=1)

    def forward(self, x):
        # x: (Batch, Time, Channels, Height, Width)
        b, t, c, h, w = x.size()
        hidden_states = []
        for i in range(self.num_layers):
            hidden_states.append(self.cell_list[i].init_hidden(b, (h, w)))
        for step in range(t):
            x_step = x[:, step, :, :, :]
            h, c = self.cell_list[0](x_step, hidden_states[0])
            hidden_states[0] = (h, c)
            current_input = h
            for i in range(1, self.num_layers):
                h, c = self.cell_list[i](current_input, hidden_states[i])
                hidden_states[i] = (h, c)
                current_input = h
        final_h = hidden_states[-1][0]
        out = self.final_conv(final_h)
        return out  # Returns logits usually, check training loss (BCEWithLogits vs BCELoss)


class HybridConvLSTMUNet(nn.Module):
    def __init__(self, in_channels, num_classes=1):
        super(HybridConvLSTMUNet, self).__init__()
        # Note: ConvLSTMCell here is slightly different (kernel_size=3 int vs tuple),
        # but the class definition above handles tuple.
        # We need to make sure we use the compatible ConvLSTMCell.
        # The Hybrid model in legacy/hybrid_model.py uses kernel_size=3.

        self.enc1 = self._conv_block(in_channels, 32)
        self.enc2 = self._conv_block(32, 64)
        self.pool = nn.MaxPool2d(2, 2)

        # LSTM at bottleneck
        # Note: Redefining ConvLSTMCell locally or adapting logic if needed.
        # For this sample, I'll assume the one above works if instantiated correctly.
        self.conv_lstm = ConvLSTMCell(
            input_dim=64, hidden_dim=64, kernel_size=(3, 3), bias=True
        )

        self.up2 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec2 = self._conv_block(64, 32)
        self.final_conv = nn.Conv2d(32, num_classes, kernel_size=1)
        self.dropout = nn.Dropout2d(p=0.3)

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
        # x: (Batch, Seq, Chan, H, W)
        b, seq, c, h, w = x.size()
        h_t, c_t = self.conv_lstm.init_hidden(b, (h // 2, w // 2))
        skip_connection = None
        for t in range(seq):
            xt = x[:, t, :, :, :]
            e1 = self.enc1(xt)
            e1_pool = self.pool(e1)
            e2 = self.enc2(e1_pool)
            if t == seq - 1:
                skip_connection = e1
            h_t, c_t = self.conv_lstm(e2, (h_t, c_t))
        h_t = self.dropout(h_t)
        d2 = self.up2(h_t)
        d2 = torch.cat([d2, skip_connection], dim=1)
        d2 = self.dec2(d2)
        out = self.final_conv(d2)
        return out


# --- 2. Inference Class ---


class FireModelInference:
    def __init__(self, model_type, weights_path, input_channels, device="cpu"):
        self.device = torch.device(device)
        self.model_type = model_type

        if model_type == "legacy_unet":
            self.model = LegacyUNet(in_channels=input_channels, out_channels=1)
        elif model_type == "convlstm":
            self.model = ConvLSTM(in_channels=input_channels, hidden_dims=[32, 32, 32])
        elif model_type == "hybrid":
            self.model = HybridConvLSTMUNet(in_channels=input_channels)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        self.model.to(self.device)
        self._load_weights(weights_path)
        self.model.eval()
        print(f"Model {model_type} loaded from {weights_path}")

    def _load_weights(self, path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Weights not found at {path}")
        try:
            # Try loading normally
            state_dict = torch.load(path, map_location=self.device)
            self.model.load_state_dict(state_dict)
        except Exception as e:
            print(f"Standard load failed, trying strict=False: {e}")
            state_dict = torch.load(path, map_location=self.device)
            self.model.load_state_dict(state_dict, strict=False)

    def predict(self, input_tensor):
        """
        input_tensor: Numpy array or Torch tensor.
        Expected shapes:
          - Legacy UNet: (Batch, Channels, Height, Width)
          - ConvLSTM/Hybrid: (Batch, Sequence_Length, Channels, Height, Width)
        """
        if isinstance(input_tensor, np.ndarray):
            input_tensor = torch.from_numpy(input_tensor).float()

        input_tensor = input_tensor.to(self.device)

        with torch.no_grad():
            output = self.model(input_tensor)

            # Apply Sigmoid if the model output is logits
            # LegacyUNet output is ALREADY sigmoid (0-1).
            # ConvLSTM output is logits (no activation in final layer).
            # Hybrid output is logits.

            if self.model_type == "legacy_unet":
                probs = output
            else:
                probs = torch.sigmoid(output)

        return probs.cpu().numpy()


# --- 3. Example Usage ---

if __name__ == "__main__":
    # Settings
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. Legacy UNet Example
    print("\n--- Testing Legacy UNet ---")
    unet_path = "weights/best_fire_unet.pth"
    if os.path.exists(unet_path):
        # Input: (Batch=1, Channels=11, H=320, W=320)
        # Note: 11 channels is standard for this project (excluding fire target)
        dummy_input = torch.randn(1, 11, 320, 320)

        predictor = FireModelInference(
            "legacy_unet", unet_path, input_channels=11, device=DEVICE
        )
        output = predictor.predict(dummy_input)
        print(f"Output Shape: {output.shape}")
        print(f"Output Range: [{output.min():.4f}, {output.max():.4f}]")
    else:
        print(f"Skipping Legacy UNet (Weights not found at {unet_path})")

    # 2. ConvLSTM Example
    print("\n--- Testing ConvLSTM ---")
    conv_path = "weights/best_convlstm.pth"
    if os.path.exists(conv_path):
        # Input: (Batch=1, Seq=3, Channels=11, H=320, W=320)
        dummy_input = torch.randn(1, 3, 11, 320, 320)

        predictor = FireModelInference(
            "convlstm", conv_path, input_channels=11, device=DEVICE
        )
        output = predictor.predict(dummy_input)
        print(f"Output Shape: {output.shape}")
        print(f"Output Range: [{output.min():.4f}, {output.max():.4f}]")
    else:
        print(f"Skipping ConvLSTM (Weights not found at {conv_path})")

    # 3. Hybrid Example
    print("\n--- Testing Hybrid Model ---")
    hybrid_path = "weights/best_hybrid_model.pth"
    if os.path.exists(hybrid_path):
        # Input: (Batch=1, Seq=3, Channels=11, H=320, W=320)
        dummy_input = torch.randn(1, 3, 11, 320, 320)

        predictor = FireModelInference(
            "hybrid", hybrid_path, input_channels=11, device=DEVICE
        )
        output = predictor.predict(dummy_input)
        print(f"Output Shape: {output.shape}")
        print(f"Output Range: [{output.min():.4f}, {output.max():.4f}]")
    else:
        print(f"Skipping Hybrid (Weights not found at {hybrid_path})")
