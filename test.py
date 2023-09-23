import os
import time

import numpy as np
import torch
import torch.utils
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from torch import nn
from torch.nn.utils import prune
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from concrete.ml.torch.compile import compile_brevitas_qat_model

from torchsummary import summary

# And some helpers for visualization.

import matplotlib.pyplot as plt

from PIL import Image

# X, y = load_digits(return_X_y=True)
path_humans = []
train_path_humans = './data/human-and-non-human/training_set/training_set/humans'
for path in os.listdir(train_path_humans):
    if '.jpg' in path:
        path_humans.append(os.path.join(train_path_humans, path))
path_nhumans = []
train_path_nhumans = './data/human-and-non-human/training_set/training_set/non-humans'
for path in os.listdir(train_path_nhumans):
    if '.jpg' in path:
        path_nhumans.append(os.path.join(train_path_nhumans, path))
len(path_nhumans), len(path_humans)

IMAGE_SIZE = 64

num_samples_humans = len(path_humans)
num_samples_nhumans = len(path_nhumans)
num_samples = num_samples_humans + num_samples_nhumans
image_size = (IMAGE_SIZE, IMAGE_SIZE, 1)
transposed_image_size = (1, IMAGE_SIZE, IMAGE_SIZE)

# Create an empty array to store the images
training_set = np.zeros((num_samples, *transposed_image_size), dtype='float32')

# Load and preprocess images
for i in range(num_samples):
    if i < num_samples_nhumans:
        path = path_nhumans[i]
    else:
        path = path_humans[i - len(path_nhumans)]
    print(path)

    img = Image.open(path).resize(image_size[:2]).convert('L')
    img_array = np.array(img)
    training_set[i] = img_array
# image = Image.fromarray(training_set[0].astype('uint8'))
# image.save('saved_image.jpg')
# Create labels (assuming 0 for non-humans and 1 for humans)
labels = np.concatenate((np.zeros(int(num_samples_nhumans)), np.ones(int(num_samples_humans))))

# The sklearn Digits data-set, though it contains digit images, keeps these images in vectors
# so we need to reshape them to 2D first. The images are 8x8 px in size and monochrome
# X = np.expand_dims(X.reshape((-1, 8, 8)), 1)
X = training_set

# nplot = 0
# fig, ax = plt.subplots(nplot, nplot, figsize=(150, 150))
# for i in range(0, nplot):
#     for j in range(0, nplot):
#         ax[i, j].imshow(X[i * nplot + j, ::].squeeze())
# plt.show()

x_train, x_test, y_train, y_test = train_test_split(training_set, labels, test_size=0.2, random_state=42)
x_train[0].shape

import brevitas.nn as qnn


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

        # First Convolution
        self.conv1 = qnn.QuantConv2d(1, 16, 5, stride=2, padding=2, weight_bit_width=w_bits)
        self.q2 = qnn.QuantIdentity(bit_width=a_bits, return_quant_tensor=True)

        # Second Convolution
        self.conv2 = qnn.QuantConv2d(16, 32, 5, stride=2, padding=2, weight_bit_width=w_bits)
        self.q3 = qnn.QuantIdentity(bit_width=a_bits, return_quant_tensor=True)

        # Fully Connected Layer
        self.fc1 = qnn.QuantLinear(32 * 16 * 16, 2, bias=True, weight_bit_width=w_bits)

        # Enable pruning, prepared for training
        # self.toggle_pruning(True)

        device = get_device()
        self.to(device)
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

        x = self.conv2(x)
        x = torch.relu(x)
        x = self.q3(x)

        x = x.flatten(1)
        x = self.fc1(x)
        return x


def set_device_for_all_layers(model, device):
    """
    Move all sub-modules of a model to the specified device.

    Args:
    - model (torch.nn.Module): The PyTorch model
    - device (torch.device): The target device ("cuda" or "cpu")
    """
    for layer in model.children():
        layer.to(device)


def get_device():
    # Check gpu availability
    # if torch.cuda.is_available():
    #     device = torch.device("cuda")
    #     print("Using GPU:", torch.cuda.get_device_name(0))
    # else:
    #     device = torch.device("cpu")
    #     print("Using CPU")
    # return device
    return torch.device("cpu")


torch.manual_seed(42)


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


# Create the tiny CNN with 10 output classes
N_EPOCHS = 150

# Create a train data loader
x_train_tensor = torch.Tensor(x_train).to(get_device())
y_train_tensor = torch.Tensor(y_train).to(get_device())
print(x_train_tensor.device)
print(y_train_tensor.device)
train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
train_dataloader = DataLoader(train_dataset, batch_size=64)

# Create a test data loader to supply batches for network evaluation (test)
x_test_tensor = torch.Tensor(x_test).to(get_device())
y_test_tensor = torch.Tensor(y_test).to(get_device())
test_dataset = TensorDataset(x_test_tensor, y_test_tensor)
test_dataloader = DataLoader(test_dataset)

nets = []
bit_range = range(2, 5)

# Train the network with Adam, output the test set accuracy every epoch
losses = []
for n_bits in bit_range:
    net = TinyCNN(2, n_bits)
    losses_bits = []
    optimizer = torch.optim.Adam(net.parameters())
    for _ in tqdm(range(N_EPOCHS), desc=f"Training with {n_bits} bit weights and activations"):
        losses_bits.append(train_one_epoch(net, optimizer, train_dataloader))
    losses.append(losses_bits)

    # Finally, disable pruning (sets the pruned weights to 0)
    # net.toggle_pruning(False)
    nets.append(net)

fig = plt.figure(figsize=(8, 4))
for losses_bits in losses:
    plt.plot(losses_bits)
plt.ylabel("Cross Entropy Loss")
plt.xlabel("Epoch")
plt.legend(list(map(str, bit_range)))
plt.title("Training set loss during training")
plt.grid(True)
plt.show()


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


for idx, net in enumerate(nets):
    test_torch(net, bit_range[idx], test_dataloader)


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


accs = []
accum_bits = []
sim_time = []

for idx in range(len(bit_range)):
    q_module = compile_brevitas_qat_model(nets[idx], x_train)

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
        f"Simulated FHE execution for {bit_range[idx]} bit network: {vl_time_bits:.2f}s, "
        f"{len(test_dataloader) / vl_time_bits:.2f}it/s"
    )

fig = plt.figure(figsize=(12, 8))
plt.rcParams["font.size"] = 14
plt.plot(bit_range, accs, "-x")
for bits, acc, accum in zip(bit_range, accs, accum_bits):
    plt.gca().annotate(str(accum), (bits - 0.1, acc + 0.01))
plt.ylabel("Accuracy on test set")
plt.xlabel("Weight & activation quantization")
plt.grid(True)
plt.title(
    "Accuracy for varying quantization bit width. Accumulator bit-width shown on graph markers"
)
plt.show()

bits_for_fhe = 3
idx_bits_fhe = bit_range.index(bits_for_fhe)

accum_bits_required = accum_bits[idx_bits_fhe]

q_module_fhe = None

net = nets[idx_bits_fhe]

q_module_fhe = compile_brevitas_qat_model(
    net,
    x_train,
)

# Generate keys first, this may take some time (up to 30min)
print("Generating keys...")
t = time.time()
q_module_fhe.fhe_circuit.keygen()
print(f"Keygen time: {time.time() - t:.2f}s")


def save_image(image_data, filename):
    # print(image_data.shape)
    # Create an Image object from the NumPy array
    image = Image.fromarray(image_data.squeeze().astype("uint8"), mode='L')
    # print(image.size)
    # Save the image to a file
    image.save(filename)  # Change 'saved_image.jpg' to your desired file name and format


def transpose_data_for_image(data):
    # Transpose the data to match the image format
    data = data.transpose(1, 2, 0)
    # Convert the data to uint8
    data = data.astype("uint8")
    # print(data.shape)
    return data


# Run inference in FHE on a single encrypted example
index = 101
mini_test_dataset = TensorDataset(torch.Tensor(x_test[[index], :]), torch.Tensor(y_test[[index]]))
mini_test_dataloader = DataLoader(mini_test_dataset)
# Save x_text[0] as an image
print(x_test[0].shape)
print(int(y_test[index]))
save_image(transpose_data_for_image(x_test[index]), "saved_image.jpg")
print(len(x_test))

print(x_test.shape)

path = "./data/franquito.jpg"
# #Open image
img = Image.open(path).resize(image_size[:2]).convert('L')
img_array = np.array(img)

test_set = np.zeros((1, *transposed_image_size), dtype='float32')
test_set[0] = img_array
print(test_set[[0]].shape)

save_image(transpose_data_for_image(test_set[0]), "saved_image.jpg")

mini_test_dataset = TensorDataset(torch.Tensor(test_set[[0]]), torch.Tensor([[1]]))
mini_test_dataloader = DataLoader(mini_test_dataset)

t = time.time()
res = test_with_concrete(
    q_module_fhe,
    mini_test_dataloader,
    use_sim=False,
)
print(f"Time per inference in FHE: {(time.time() - t) / len(mini_test_dataset):.2f}")
print(f"Accuracy in FHE: {res:.2f}")
