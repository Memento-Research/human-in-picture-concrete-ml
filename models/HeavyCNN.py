import torch
from torch import nn


class HeavyCNN(nn.Module):

    def __init__(self, n_classes) -> None:
        """Construct the CNN with a configurable number of classes."""
        super().__init__()

        # Single Convolution
        self.conv1 = nn.Conv2d(1, 16, 5, stride=2, padding=2)
        self.conv2 = nn.Conv2d(16, 32, 5, stride=2, padding=2)

        # Fully Connected Layer
        self.fc1 = nn.Linear(32 * 16 * 16, n_classes)

    def forward(self, x):
        """Run inference on the modified CNN, apply the decision layer on the reshaped conv output."""
        x = self.conv1(x)
        x = torch.relu(x)
        x = self.conv2(x)
        x = torch.relu(x)
        x = x.flatten(1)
        x = self.fc1(x)
        return x

