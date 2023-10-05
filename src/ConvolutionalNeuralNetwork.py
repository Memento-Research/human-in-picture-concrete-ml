import time
import torch.utils
import numpy as np
from concrete.fhe import Configuration

from utils.results_utils import export_times
from utils.dataset_utils import get_loaded_dataset, create_dataloader, process_images, save_image
from utils.cnn_utils import create_network, train_network, test_network
from utils.fhe_utils import compile_network, test_quantized_module, test_with_concrete
from utils.graphics_utils import plot_training_loss
from utils.arguments_utils import parse_arguments

from models.TinyCNN import TinyCNN

required_args = {
    "IMAGE_SIZE": None,
}


# ARGS: IMAGE_SIZE
def main():
    # Parse arguments
    args = parse_arguments(required_args)
    print(args)
    image_size = int(args["IMAGE_SIZE"])

    images_to_load = 250  # Load 100 images from each class

    humans_path = './data/human-and-non-human/training_set/training_set/humans'
    not_humans_path = './data/human-and-non-human/training_set/training_set/non-humans'

    # Load the dataset
    x_train, x_test, y_train, y_test = get_loaded_dataset(
        humans_path,
        not_humans_path,
        image_size,
        images_to_load,
    )

    train_dataloader = create_dataloader(x_train, y_train, 64)
    test_dataloader = create_dataloader(x_test, y_test, 1)

    n_classes = 2  # humans and not humans
    n_bits = 6  # Quantization bit-width
    p_error = 0.1  # Probability of error

    net = create_network(TinyCNN, image_size, n_classes)

    torch.manual_seed(42)

    # Epochs to train for
    n_epochs = 150

    # Train the network
    losses = train_network(net, n_epochs, train_dataloader)

    # Plot the cross-entropy loss for each epoch
    plot_training_loss(losses)

    # Test the network in fp32
    test_network(net, n_bits, test_dataloader)

    # Compile the network with Concrete ML
    print("Compiling network with Concrete ML...")
    configuration = Configuration(show_graph=False,
                                  show_statistics=False,
                                  show_progress=True)
    q_module_fhe = compile_network(net, x_train, n_bits, p_error, verbose=False, configuration=configuration)

    # Test the network in FHE using simulation
    print("Testing in FHE using simulation FHE execution...")
    test_quantized_module(q_module_fhe, n_bits, test_dataloader, use_sim=True)

    # Generate keys first, this may take some time (up to 30min)
    print("Generating key...")
    t = time.time()
    q_module_fhe.fhe_circuit.keygen()
    print(f"Keygen time: {time.time() - t:.2f}s")

    # Run inference in FHE on n encrypted examples
    n = 10
    x = np.array(x_test[:n])
    y = np.array(y_test[:n])
    mini_test_dataloader = create_dataloader(x, y, 1)

    # path = "./data/white.jpg"
    #
    # img = process_images([path], IMAGE_SIZE, 1)[0]
    # x = np.array([img])
    # y = np.array([[0]])
    # mini_test_dataloader = create_dataloader(x, y, 1)
    #
    # save_image(img, "saved_image.jpg")

    # Test the network in FHE using real FHE execution
    print("Testing in FHE using real FHE execution...")
    t = time.time()
    res, times = test_with_concrete(
        q_module_fhe,
        mini_test_dataloader,
        use_sim=False,
    )
    print(f"Time per inference in FHE: {(time.time() - t) / len(mini_test_dataloader):.2f}")
    print(f"Accuracy in FHE: {res:.2f}%")

    export_times(image_size, times)


if __name__ == "__main__":
    main()
