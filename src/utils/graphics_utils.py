import os

import matplotlib.pyplot as plt
import numpy as np


def plot_training_loss(losses_bits):
    fig = plt.figure(figsize=(8, 4))
    plt.plot(losses_bits)
    plt.ylabel("Cross Entropy Loss")
    plt.xlabel("Epoch")
    plt.title("Training set loss during training")
    plt.grid(True)
    plt.show()


def plot_times(directory_path: str):
    # Initialize empty lists to store data from all files
    image_sizes = []
    average_values = []
    std_deviations = []

    # Loop through all files in the folder
    for filename in os.listdir(directory_path):
        if filename.endswith('.txt'):
            # Construct the full file path
            file_path = os.path.join(directory_path, filename)

            # Load the data from the current file
            with open(file_path, 'r') as file:
                lines = file.readlines()

            # Extract image size and values
            image_size = int(lines[0].strip().split(': ')[1])
            data_values = [float(line.strip()) for line in lines[1:]]

            # Calculate the average and standard deviation
            average_value = np.mean(data_values)
            std_deviation = np.std(data_values)

            # Append the values to the respective lists
            image_sizes.append(image_size * image_size)
            average_values.append(average_value)
            std_deviations.append(std_deviation)

    # Sort the data points based on image_size
    sorted_data = sorted(zip(image_sizes, average_values, std_deviations))

    # Unzip the sorted data back into separate lists
    image_sizes, average_values, std_deviations = zip(*sorted_data)

    # Create a line plot with lines connecting the data points
    plt.figure(figsize=(8, 6))
    plt.errorbar(image_sizes, average_values, yerr=std_deviations, fmt='-o', label='Average with Std Deviation',
                 markersize=5)
    plt.xlabel('Image Size [px]')
    plt.ylabel('Time [s]')
    plt.title('Average and Standard Deviation for Multiple Files')
    plt.legend()
    plt.grid(True)

    # Show the plot or save it to a file
    # plt.show()
    # Uncomment the next line to save the plot as an image file (e.g., PNG):
    plt.savefig('./outputs/average_std_deviation_plot.png')
