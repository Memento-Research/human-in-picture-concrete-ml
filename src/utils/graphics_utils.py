import os

import matplotlib.pyplot as plt
import numpy as np


class Data:
    FILE_OFFSET = 7

    image_size: int
    n_bits: float
    p_error: float
    clear_precision: float
    fhe_precision: float
    values: [float]

    def __init__(self, lines: [str]):
        self.image_size = int(lines[0].strip().split(': ')[1])
        self.n_bits = float(lines[1].strip().split(': ')[1])
        self.p_error = float(lines[2].strip().split(': ')[1])
        self.clear_precision = float(lines[3].strip().split(': ')[1])
        self.fhe_precision = float(lines[4].strip().split(': ')[1])
        self.values = [float(line.strip()) for line in lines[self.FILE_OFFSET:]]


def plot_training_loss(losses_bits):
    fig = plt.figure(figsize=(8, 4))
    plt.plot(losses_bits)
    plt.ylabel("Cross Entropy Loss")
    plt.xlabel("Epoch")
    plt.title("Training set loss during training")
    plt.grid(True)
    plt.show()


def plot_entropy_loss(directory_path: str, filenames: [str]):
    data: [Data] = load_files(directory_path, filenames)

    # Plot the loss evolution
    plt.figure(figsize=(10, 6))
    for (i, d) in enumerate(data):
        plt.plot(d.values, label=f"{d.image_size}x{d.image_size} px ({d.n_bits} bits, {d.p_error} p_error)")

    plt.title("Loss Evolution for CNN with Different Image Sizes")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig('./outputs/loss_evolution_plot.png')


def plot_times(directory_path: str, filenames: [str]):
    # Initialize empty lists to store data from all files
    image_labels = []
    image_sizes = []
    average_values = []
    std_deviations = []

    # Loop through all files in the folder
    for filename in filenames:
        # Construct the full file path
        file_path = os.path.join(directory_path, filename)

        # Load the data from the current file
        with open(file_path, 'r') as file:
            lines = file.readlines()

        # Extract image size and values
        data = Data(lines)
        image_size = data.image_size
        data_values = data.values

        # Calculate the average and standard deviation
        average_value = np.mean(data_values)
        std_deviation = np.std(data_values)

        # Append the values to the respective lists
        image_labels.append(f"{image_size}x{image_size}")
        image_sizes.append(image_size * image_size)
        average_values.append(average_value)
        std_deviations.append(std_deviation)

    # Sort the data points based on image_size
    sorted_data = sorted(zip(image_sizes, image_labels, average_values, std_deviations))

    # Unzip the sorted data back into separate lists
    image_sizes, image_labels, average_values, std_deviations = zip(*sorted_data)

    # Create a line plot with lines connecting the data points
    plt.figure(figsize=(8, 6))
    plt.errorbar(image_sizes, average_values, yerr=std_deviations, fmt='-o', label='Average with Std Deviation',
                 markersize=5)
    plt.xticks(image_sizes, image_labels)
    plt.xlabel('Image Size [px]')
    plt.ylabel('Time [s]')
    plt.title('Average and Standard Deviation for Multiple Files')
    plt.legend()
    plt.grid(True)

    # Show the plot or save it to a file
    # plt.show()
    # Uncomment the next line to save the plot as an image file (e.g., PNG):
    plt.savefig('./outputs/average_std_deviation_plot.png')


def plot_points(directory_path: str, filenames: [str], title: str, x_label: str, y_label: str):
    data: [Data] = load_files(directory_path, filenames)

    x_values: [float] = [(data[i].n_bits if x_label == "n_bits" else data[i].p_error) for i in range(len(data))]
    clear_precisions: [float] = [data[i].clear_precision for i in range(len(data))]
    fhe_precisions: [float] = [data[i].fhe_precision for i in range(len(data))]

    # sort the data points based on x_values
    sorted_data = sorted(zip(x_values, clear_precisions, fhe_precisions))
    x_values, clear_precisions, fhe_precisions = zip(*sorted_data)

    plt.figure(figsize=(10, 6))
    plt.plot(x_values, clear_precisions, '-o', label="Clear Precision")
    plt.plot(x_values, fhe_precisions, '-o', label="FHE Precision")
    plt.title(f"{title} for ${data[0].image_size}x{data[0].image_size}$ px")
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    plt.grid(True)
    plt.savefig(f'./outputs/{x_label}_precision_plot.png')


def load_files(directory_path: str, filenames: [str]):
    data: [Data] = []
    for filename in filenames:
        file_path = os.path.join(directory_path, filename)
        with open(file_path, 'r') as file:
            lines = file.readlines()
            data.append(Data(lines))
    return data
