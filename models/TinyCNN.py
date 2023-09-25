import brevitas.nn as qnn
import torch
from torch import nn


class TinyCNN(nn.Module):
    """A very small CNN to classify the sklearn digits data-set.

    This class also allows pruning to a maximum of 10 active neurons, which
    should help keep the accumulator bit width low.
    """

    def __init__(self, n_classes, n_bits) -> None:
        """Construct the CNN with a configurable number of classes."""
        super().__init__()

        a_bits = n_bits
        w_bits = n_bits

        # This network has a total complexity of 1216 MAC
        # Quantization Layer
        self.q1 = qnn.QuantIdentity(bit_width=a_bits, return_quant_tensor=True)

        # Single Convolution
        self.conv1 = qnn.QuantConv2d(1, 16, 5, stride=4, padding=2, weight_bit_width=w_bits)
        self.q2 = qnn.QuantIdentity(bit_width=a_bits, return_quant_tensor=True)

        # Fully Connected Layer
        self.fc1 = qnn.QuantLinear(16 * 16 * 16, 2, bias=True, weight_bit_width=w_bits)

        # Enable pruning, prepared for training
        # self.toggle_pruning(True)

        # device = get_device()
        # self.to(device)
        # Send all layers to the device
        # self.q1.to(device)
        # self.conv1.to(device)
        # self.q2.to(device)
        # self.fc1.to(device)

    # def toggle_pruning(self, enable):
    #     """Enables or removes pruning."""
    #
    #     # Maximum number of active neurons (i.e., corresponding weight != 0)
    #     n_active = 12
    #
    #     # Go through all the convolution layers
    #     for layer in [self.conv1]:
    #         s = layer.weight.shape
    #
    #         # Compute fan-in (number of inputs to a neuron)
    #         # and fan-out (number of neurons in the layer)
    #         st = [s[0], np.prod(s[1:])]
    #
    #         # The number of input neurons (fan-in) is the product of
    #         # the kernel width x height x inChannels.
    #         if st[1] > n_active:
    #             if enable:
    #                 # This will create a forward hook to create a mask tensor that is multiplied
    #                 # with the weights during forward. The mask will contain 0s or 1s
    #                 prune.l1_unstructured(layer, "weight", (st[1] - n_active) * st[0])
    #             else:
    #                 # When disabling pruning, the mask is multiplied with the weights
    #                 # and the result is stored in the weights member
    #                 prune.remove(layer, "weight")

    def forward(self, x):
        """Run inference on the modified CNN, apply the decision layer on the reshaped conv output."""
        x = self.q1(x)
        x = self.conv1(x)
        x = torch.relu(x)
        x = self.q2(x)

        x = x.flatten(1)
        x = self.fc1(x)
        return x