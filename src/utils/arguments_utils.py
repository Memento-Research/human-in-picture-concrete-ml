import sys


def parse_arguments(required_args: dict):
    if len(sys.argv) - 1 != len(required_args):
        # Print the name of each argument
        print(f"Usage: python3 ConvolutionalNeuralNetwork.py <{' '.join(required_args.keys())}>")
        exit(1)

    # Populate the arguments as values for each key
    args = {}
    for i, key in enumerate(required_args.keys()):
        args[key] = sys.argv[i + 1]
    return args
