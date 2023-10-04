import time

import numpy as np
from concrete.fhe import Configuration
from concrete.ml.torch.compile import compile_torch_model
from tqdm import tqdm

from utils.time_utils import log_time


@log_time
def compile_network(net, x_train, n_bits: int, p_error: float, verbose=False, configuration: Configuration = None):
    """Compile a network with Concrete ML."""
    q_module = compile_torch_model(net,
                                   x_train,
                                   rounding_threshold_bits=n_bits,
                                   p_error=p_error,
                                   verbose=verbose,
                                   configuration=configuration
                                   )
    return q_module


def test_quantized_module(quantized_module, n_bits, test_dataloader, use_sim):
    acc, times = test_with_concrete(
        quantized_module,
        test_dataloader,
        use_sim=use_sim
    )

    print(f"{'Simulated' if use_sim else 'Real'} FHE execution for {n_bits} bit network accuracy: {acc:.2f}%")


# Concrete ML testing function
@DeprecationWarning
def compile_and_test_with_concrete(net, x_train, test_dataloader):
    n_bits = 6

    q_module = compile_torch_model(net, x_train, rounding_threshold_bits=n_bits, p_error=0.1)

    start_time = time.time()
    acc = test_with_concrete(
        q_module,
        test_dataloader,
        use_sim=True
    )
    sim_time = time.time() - start_time

    print(f"Simulated FHE execution for {n_bits} bit network accuracy: {acc:.2f}% takes {sim_time:.2f}s")

    return q_module, acc


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

    times = []

    # Iterate over the test batches and accumulate predictions and ground truth labels in a vector
    idx = 0
    for data, target in tqdm(test_loader):
        start_time = time.time()

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

        # measure time
        times.append(time.time() - start_time)

    # Compute and report results
    n_correct = np.sum(all_targets == all_y_pred)

    return (n_correct / len(test_loader)) * 100, times
