import time
import torch
import numpy as np
from torch import nn
from tqdm import tqdm
from concrete.ml.torch.compile import compile_brevitas_qat_model


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