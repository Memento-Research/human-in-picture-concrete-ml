import matplotlib.pyplot as plt


def plot_training_loss(bits_range, losses):
    fig = plt.figure(figsize=(8, 4))
    for losses_bits in losses:
        plt.plot(losses_bits)
    plt.ylabel("Cross Entropy Loss")
    plt.xlabel("Epoch")
    plt.legend(list(map(str, bits_range)))
    plt.title("Training set loss during training")
    plt.grid(True)
    plt.show()


def plot_accuracy(bits_range, accuracies, accum_bits):
    fig = plt.figure(figsize=(12, 8))
    plt.rcParams["font.size"] = 14
    plt.plot(bits_range, accuracies, "-x")
    for bits, acc, accum in zip(bits_range, accuracies, accum_bits):
        plt.gca().annotate(str(accum), (bits - 0.1, acc + 0.01))
    plt.ylabel("Accuracy on test set")
    plt.xlabel("Weight & activation quantization")
    plt.grid(True)
    plt.title(
        "Accuracy for varying quantization bit width. Accumulator bit-width shown on graph markers"
    )
    plt.show()
