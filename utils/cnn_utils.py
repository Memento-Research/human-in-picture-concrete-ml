import numpy as np
import torch
from torch import nn
from tqdm import tqdm


# Create Network
# Receive network class and instantiate it
def create_network(network_class, image_size, n_classes):
    return network_class(image_size, n_classes)


# Train the CNN
# Train the network with Adam, output the test set accuracy every epoch
def train_network(net, epochs, train_dataloader):
    losses_bits = []
    optimizer = torch.optim.Adam(net.parameters())
    for _ in tqdm(range(epochs), desc=f"Training"):
        losses_bits.append(train_one_epoch(net, optimizer, train_dataloader))

    return losses_bits


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
# No se usa en el notebook. Puede ser util para mas adelante
@DeprecationWarning
def test_networks(nets, bits_range, test_dataloader):
    for net, n_bits in zip(nets, bits_range):
        test_network(net, n_bits, test_dataloader)


def test_network(net, n_bits, test_loader):
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
