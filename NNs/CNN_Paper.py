import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    """
    Expects input shaped (B, 11, 61, 1) [channels-last].
    Internally permutes to (B, C=1, H=11, W=61) for Conv2d.
    """
    def __init__(self, num_classes: int = 3):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)  # "same" padding
        self.relu = nn.ReLU(inplace=True)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(32 * 11 * 61, 64)  # after "same" conv: 32x11x61
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        self.out = nn.Linear(64, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 11, 61, 1) -> (B, 1, 11, 61)
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.relu(self.conv(x))
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = F.softmax(self.out(x), dim=1)  # For training with CrossEntropyLoss, return logits instead.
        return x

if __name__ == "__main__":
    model = Net()
    dummy = torch.randn(2, 11, 61, 1)  # batch of 2
    out = model(dummy)
    print(out.shape)  # torch.Size([2, 3])