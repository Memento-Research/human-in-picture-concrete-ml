import torch
from torch import nn


class HeavyCNN(nn.Module):
        def __init__(self, n_classes) -> None:
            """Construct the CNN with a configurable number of classes."""
            super().__init__()

            self.conv1 = nn.Conv2d(1, 32, 3, stride=1, padding=1)  # output: 32x64x64
            self.conv2 = nn.Conv2d(32, 64, 3, stride=2, padding=1)  # output: 64x32x32
            self.conv3 = nn.Conv2d(64, 128, 3, stride=2, padding=1)  # output: 128x16x16

            self.fc1 = nn.Linear(128 * 16 * 16, 1024)
            self.fc2 = nn.Linear(1024, n_classes)

        def forward(self, x):
            """Run inference on the 3-layer CNN."""
            x = torch.relu(self.conv1(x))
            x = torch.relu(self.conv2(x))
            x = torch.relu(self.conv3(x))

            x = x.view(x.size(0), -1)  # flatten
            x = torch.relu(self.fc1(x))
            x = self.fc2(x)
            return x

