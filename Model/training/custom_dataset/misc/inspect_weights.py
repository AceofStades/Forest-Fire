import torch

path = "Model/checkouts/best_unet.pth"
state_dict = torch.load(path, map_location="cpu", weights_only=True)
print("Bias in saved state dict:")
print(state_dict["outc.bias"])

print("Conv weights max:")
print(state_dict["outc.weight"].max())
