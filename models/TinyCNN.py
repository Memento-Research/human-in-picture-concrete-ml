import torch
from torch import nn


class TinyCNN(nn.Module):
    """A very small CNN to classify the sklearn digits data-set.

    This class also allows pruning to a maximum of 10 active neurons, which
    should help keep the accumulator bit width low.
    """

    def __init__(self, n_classes) -> None:
        """Construct the CNN with a configurable number of classes."""
        super().__init__()

        # Single Convolution
        self.conv1 = nn.Conv2d(1, 16, 5, stride=4, padding=2)

        # Fully Connected Layer
        self.fc1 = nn.Linear(16 * 16 * 16, n_classes)

    def forward(self, x):
        """Run inference on the modified CNN, apply the decision layer on the reshaped conv output."""
        x = self.conv1(x)
        x = torch.relu(x)
        x = x.flatten(1)
        x = self.fc1(x)
        return x
