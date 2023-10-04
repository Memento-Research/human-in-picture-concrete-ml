import matplotlib.pyplot as plt


def plot_training_loss(losses_bits):
    fig = plt.figure(figsize=(8, 4))
    plt.plot(losses_bits)
    plt.ylabel("Cross Entropy Loss")
    plt.xlabel("Epoch")
    plt.title("Training set loss during training")
    plt.grid(True)
    plt.show()
