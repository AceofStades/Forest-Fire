import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from data_utils import load_split_data


def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
    )


class UNet(nn.Module):
    def __init__(self, in_channels, out_channels=1):
        super().__init__()

        # Encoder (Downsampling Path)
        self.dconv_down1 = double_conv(in_channels, 64)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dconv_down2 = double_conv(64, 128)
        self.dconv_down3 = double_conv(128, 256)
        self.dconv_down4 = double_conv(256, 512)

        # Bottleneck
        self.dconv_bottom = double_conv(512, 1024)

        # Decoder (Upsampling Path and Skip Connections)
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
        # Encoder
        conv1 = self.dconv_down1(x)
        x = self.maxpool(conv1)
        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)
        conv3 = self.dconv_down3(x)
        x = self.maxpool(conv3)
        conv4 = self.dconv_down4(x)
        x = self.maxpool(conv4)

        # Bottleneck
        u = self.dconv_bottom(x)

        # Decoder with Skip Connections
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


def train_model(train_loader, val_loader, input_channels):
    NUM_EPOCHS = 25
    LEARNING_RATE = 1e-4
    DEVICE = torch.device("cuda")
    print("Training on device: ", DEVICE)

    model = UNet(in_channels=input_channels, out_channels=1).to(DEVICE)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    best_val_loss = float("inf")

    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0

        train_pbar = tqdm(
            train_loader, desc=f"Epoch {epoch + 1}/{NUM_EPOCHS} [TRAIN]", unit="batch"
        )

        for inputs, labels in train_pbar:
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            train_pbar.set_postfix(loss=f"{loss.item():.6f}")

        epoch_loss = running_loss / len(train_loader.dataset)

        # Validation
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
        	val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [VALID]", unit="batch")
         	for inputs, labels in val_pbar:
          		inputs = inputs.to(DEVICE)
            	labels = labels.to(DEVICE)
