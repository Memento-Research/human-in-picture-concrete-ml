import torch


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
