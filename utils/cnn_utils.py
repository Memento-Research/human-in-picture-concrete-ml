import time
import torch
import numpy as np
from torch import nn
from tqdm import tqdm
from concrete.ml.torch.compile import compile_brevitas_qat_model

from utils.dataset_utils import create_dataloader
from models.TinyCNN import TinyCNN


## Train the CNN
# Train the network with Adam, output the test set accuracy every epoch
def create_and_train_networks(bits_range, epochs, train_dataloader):
    nets = []
    losses = []

    for n_bits in bits_range:
        net = TinyCNN(2, n_bits)
        losses_bits = []
        optimizer = torch.optim.Adam(net.parameters())
        for _ in tqdm(range(epochs), desc=f"Training with {n_bits} bit weights and activations"):
            losses_bits.append(train_one_epoch(net, optimizer, train_dataloader))
        losses.append(losses_bits)

        # Finally, disable pruning (sets the pruned weights to 0)
        # net.toggle_pruning(False)
        nets.append(net)

    return nets, losses

def train_one_epoch(net, optimizer, train_loader):
    # Cross Entropy loss for classification when not using a softmax layer in the network
    loss = nn.CrossEntropyLoss()

    net.train()
    avg_loss = 0
    for data, target in train_loader:
        optimizer.zero_grad()
        output = net(data)
        loss_net = loss(output, target.long())
        loss_net.backward()
        optimizer.step()
        avg_loss += loss_net.item()

    return avg_loss / len(train_loader)


# Test the network on the test set
def test_networks(nets, bits_range, test_dataloader):
    for net, n_bits in zip(nets, bits_range):
        test_torch(net, n_bits, test_dataloader)


def test_torch(net, n_bits, test_loader):
    """Test the network: measure accuracy on the test set."""

    # Freeze normalization layers
    net.eval()

    all_y_pred = np.zeros((len(test_loader)), dtype=np.int64)
    all_targets = np.zeros((len(test_loader)), dtype=np.int64)

    # Iterate over the batches
    idx = 0
    for data, target in test_loader:
        # Accumulate the ground truth labels
        endidx = idx + target.shape[0]
        all_targets[idx:endidx] = target.numpy()

        # Run forward and get the predicted class id
        output = net(data).argmax(1).detach().numpy()
        all_y_pred[idx:endidx] = output

        idx += target.shape[0]

    # Print out the accuracy as a percentage
    n_correct = np.sum(all_targets == all_y_pred)
    print(
        f"Test accuracy for {n_bits}-bit weights and activations: "
        f"{n_correct / len(test_loader) * 100:.2f}%"
    )


# Concrete ML testing function
def test_networks_with_concrete(nets, bits_range, x_train, test_dataloader):
    accs = []
    accum_bits = []
    sim_time = []

    for net, n_bits in zip(nets, bits_range):
        q_module = compile_brevitas_qat_model(net, x_train)

        accum_bits.append(q_module.fhe_circuit.graph.maximum_integer_bit_width())

        start_time = time.time()
        accs.append(
            test_with_concrete(
                q_module,
                test_dataloader,
                use_sim=True,
            )
        )
        sim_time.append(time.time() - start_time)

    for idx, vl_time_bits in enumerate(sim_time):
        print(
            f"Simulated FHE execution for {bits_range[idx]} bit network: {vl_time_bits:.2f}s, "
            f"{len(test_dataloader) / vl_time_bits:.2f}it/s"
        )

    return accs, accum_bits


# We introduce the `test_with_concrete` function which allows us to test a Concrete ML model in one of two modes:
# - in FHE
# - in the clear, using simulated FHE execution
#
# Note that it is trivial to toggle between the two modes.
def test_with_concrete(quantized_module, test_loader, use_sim):
    """Test a neural network that is quantized and compiled with Concrete ML."""

    # Casting the inputs into int64 is recommended
    all_y_pred = np.zeros((len(test_loader)), dtype=np.int64)
    all_targets = np.zeros((len(test_loader)), dtype=np.int64)

    # Iterate over the test batches and accumulate predictions and ground truth labels in a vector
    idx = 0
    for data, target in tqdm(test_loader):
        data = data.numpy()
        target = target.numpy()

        fhe_mode = "simulate" if use_sim else "execute"

        # Quantize the inputs and cast to appropriate data type
        y_pred = quantized_module.forward(data, fhe=fhe_mode)

        endidx = idx + target.shape[0]

        # Accumulate the ground truth labels
        all_targets[idx:endidx] = target

        # Get the predicted class id and accumulate the predictions
        y_pred = np.argmax(y_pred, axis=1)
        all_y_pred[idx:endidx] = y_pred

        # Update the index
        idx += target.shape[0]

    # Compute and report results
    n_correct = np.sum(all_targets == all_y_pred)
    print(all_targets)
    print(all_y_pred)

    return n_correct / len(test_loader)
