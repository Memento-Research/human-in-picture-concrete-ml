import os

import matplotlib.pyplot as plt
import numpy as np


# TODO REFACTOR THIS FILE

class Data:
    FILE_OFFSET = 7

    image_size: int
    n_bits: int
    p_error: float
    clear_precision: float
    fhe_precision: float
    values: [float]

    def __init__(self, lines: [str]):
        self.image_size = int(lines[0].strip().split(': ')[1])
        self.n_bits = int(lines[1].strip().split(': ')[1])
        self.p_error = float(lines[2].strip().split(': ')[1]) / 100
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


def plot_entropy_loss(directory_path: str):
    data: [Data] = load_files(directory_path)

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


def plot_times(directory_path: str):
    # Initialize empty lists to store data from all files
    image_labels = []
    image_sizes = []
    average_values = []
    std_deviations = []

    # Loop through all files in the folder
    for filename in os.listdir(directory_path):
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


def plot_points(directory_path: str, title: str, x_label: str, y_label: str):
    data: [Data] = load_files(directory_path)

    grouped_data = {}
    # If x_label is "n_bits" group the data by n_bits
    if x_label == "n_bits":
        for d in data:
            if d.n_bits not in grouped_data:
                grouped_data[d.n_bits] = []
            grouped_data[d.n_bits].append(d)
    # If x_label is "p_error" group the data by p_error
    elif x_label == "p_error":
        for d in data:
            if d.p_error not in grouped_data:
                grouped_data[d.p_error] = []
            grouped_data[d.p_error].append(d)

    x_values: [float] = [key for key in grouped_data.keys()]
    # For each group calculate the average and standard deviation
    clear_precisions: [float] = [np.mean([d.clear_precision for d in grouped_data[key]]) for key in grouped_data.keys()]
    cleat_std_deviations: [float] = [np.std([d.clear_precision for d in grouped_data[key]]) for key in
                                     grouped_data.keys()]
    fhe_precisions: [float] = [np.mean([d.fhe_precision for d in grouped_data[key]]) for key in grouped_data.keys()]
    fhe_std_deviations: [float] = [np.std([d.fhe_precision for d in grouped_data[key]]) for key in grouped_data.keys()]

    # sort the data points based on x_values
    sorted_data = sorted(zip(x_values, clear_precisions, fhe_precisions))
    x_values, clear_precisions, fhe_precisions = zip(*sorted_data)

    plt.figure(figsize=(10, 6))
    plt.errorbar(x_values, clear_precisions, yerr=cleat_std_deviations, fmt='-o',
                 label='Clear Precision with Std Deviation')
    plt.errorbar(x_values, fhe_precisions, yerr=fhe_std_deviations, fmt='-o', label='FHE Precision with Std Deviation')
    plt.title(f"{title} for ${data[0].image_size}x{data[0].image_size}$ px")
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    plt.xticks(x_values)
    plt.grid(True)
    plt.savefig(f'./outputs/{x_label}_precision_plot.png')


def plot_times_for_n_bits(directory_path: str):
    data: [Data] = load_files(directory_path)

    grouped_data = {}
    for d in data:
        if d.n_bits not in grouped_data:
            grouped_data[d.n_bits] = []
        grouped_data[d.n_bits].append(d)

    x_values: [float] = [key for key in grouped_data.keys()]
    # For each group calculate the average and standard deviation
    fhe_times: [float] = [np.mean([d.values for d in grouped_data[key]]) for key in grouped_data.keys()]
    std_deviation: [float] = [np.std([d.values for d in grouped_data[key]]) for key in grouped_data.keys()]

    # sort the data points based on x_values
    sorted_data = sorted(zip(x_values, fhe_times, std_deviation))
    x_values, fhe_times, std_deviation = zip(*sorted_data)

    plt.figure(figsize=(10, 6))
    plt.errorbar(x_values, fhe_times, yerr=std_deviation, fmt='-o', label='FHE Time with Std Deviation')
    plt.title(f"Time for ${data[0].image_size}x{data[0].image_size}$ px")
    plt.xlabel("N_bits")
    plt.ylabel("Time [s]")
    plt.xticks(x_values)
    plt.legend()
    plt.grid(True)
    plt.savefig('./outputs/n_bits_time_plot.png')


def plot_times_for_p_error(directory_path: str):
    data: [Data] = load_files(directory_path)

    grouped_data = {}
    for d in data:
        if d.p_error not in grouped_data:
            grouped_data[d.p_error] = []
        grouped_data[d.p_error].append(d)

    x_values: [float] = [key for key in grouped_data.keys()]
    # For each group calculate the average and standard deviation
    fhe_times: [float] = [np.mean([d.values for d in grouped_data[key]]) for key in grouped_data.keys()]
    std_deviation: [float] = [np.std([d.values for d in grouped_data[key]]) for key in grouped_data.keys()]

    # sort the data points based on x_values
    sorted_data = sorted(zip(x_values, fhe_times, std_deviation))
    x_values, fhe_times, std_deviation = zip(*sorted_data)

    plt.figure(figsize=(10, 6))
    plt.errorbar(x_values, fhe_times, yerr=std_deviation, fmt='-o', label='FHE Time with Std Deviation')
    plt.title(f"Time for ${data[0].image_size}x{data[0].image_size}$ px")
    plt.xlabel("P_error")
    plt.ylabel("Time [s]")
    plt.xticks(x_values)
    plt.legend()
    plt.grid(True)
    plt.savefig('./outputs/p_error_time_plot.png')


def load_files(directory_path):
    data: [Data] = []
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        with open(file_path, 'r') as file:
            lines = file.readlines()
            data.append(Data(lines))
    return data
