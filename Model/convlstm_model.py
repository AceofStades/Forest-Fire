import torch
import torch.nn as nn
import torch.nn.functional as F


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

        # Concatenate along channel axis
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

        # Stack of ConvLSTM Cells
        cell_list = []

        # Layer 1
        cell_list.append(
            ConvLSTMCell(
                input_dim=in_channels,
                hidden_dim=hidden_dims[0],
                kernel_size=(3, 3),
                bias=True,
            )
        )

        # Layers 2 to N
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

        # Final prediction layer (Map hidden state -> Fire Probability)
        self.final_conv = nn.Conv2d(hidden_dims[-1], out_channels, kernel_size=1)

    def forward(self, x):
        """
        x: (Batch, Time, Channels, Height, Width)
        """
        b, t, c, h, w = x.size()

        # Initialize hidden states for each layer
        hidden_states = []
        for i in range(self.num_layers):
            hidden_states.append(self.cell_list[i].init_hidden(b, (h, w)))

        # Loop through time steps
        for step in range(t):
            x_step = x[:, step, :, :, :]

            # Layer 1
            h, c = self.cell_list[0](x_step, hidden_states[0])
            hidden_states[0] = (h, c)

            # Pass output of Layer 1 to Layer 2, etc.
            current_input = h
            for i in range(1, self.num_layers):
                h, c = self.cell_list[i](current_input, hidden_states[i])
                hidden_states[i] = (h, c)
                current_input = h

        # We only care about the prediction at the FINAL time step
        # Take the hidden state of the last layer at the last time step
        final_h = hidden_states[-1][0]

        # 1x1 Conv to get probability map
        out = self.final_conv(final_h)
        return out
