import torch
from torch import nn


class TinyCNN(nn.Module):
    """A very small CNN to classify the sklearn digits data-set.

    This class also allows pruning to a maximum of 10 active neurons, which
    should help keep the accumulator bit width low.
    """

    def __init__(self, image_size, n_classes) -> None:
        """Construct the CNN with a configurable number of classes."""
        super().__init__()

        # Single Convolution
        kernel_size, stride, padding = 5, 4, 2
        self.conv1 = nn.Conv2d(1, 16, kernel_size, stride=stride, padding=padding)

        feature_map_size = self.get_feature_map_size(image_size, kernel_size, stride, padding)

        # Fully Connected Layer
        self.fc1 = nn.Linear(16 * (feature_map_size ** 2), n_classes)

    def forward(self, x):
        """Run inference on the modified CNN, apply the decision layer on the reshaped conv output."""
        x = self.conv1(x)
        x = torch.relu(x)
        x = x.flatten(1)
        x = self.fc1(x)
        return x

    @staticmethod
    def get_feature_map_size(image_size, kernel_size, stride, padding):
        """Calculate the feature map size after a convolution."""
        return (image_size - kernel_size + 2 * padding) // stride + 1
