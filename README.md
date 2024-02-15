# Human In Picture (HIP) using Concrete ML

## Objective

- Create a CNN using Concrete ML to identify if there is a human in a picture.
- Identify the performance limits of a model for this problem using Concrete ML.
- Create benchmarks for different sizes of the input.

## Requirements

### Kaggle account

- You will need to have a Kaggle account in order to download the dataset, we recommend using Google Login.
- After creating an account you will need to download a `kaggle.json` file as the API key.
- You can find that file by going to [Your Profile](https://www.kaggle.com/settings/account) and scrolling down to
  the `API` section.
- Then create a new token and you will download the `kaggle.json`.
- Create a new hidden dictory called `.kaggle` in the home directory.
- Move `kaggle.json` to `.kaggle`

### Python version

- Required Python version: [3.10](https://www.python.org/downloads/release/python-3100/) < 3.11
- This project uses Poetry. If you don't already have Poetry installed, `make deps` will install it for you.
    - Make sure to have installed `curl` before running the command.

## Installation

### Dependencies

- To install dependencies you will need to run: 

```sh
make deps
```

### Dataset

- To download the dataset you will need to run:

```sh
make data
```

## Running the project

### Single run

- To run the project only once you'll need to run:

```bash
make run
```

### Benchmarks

- To run the benchmarks you'll need to run:

```bash
make benchmark
```

- This will run the benchmarks for the following input sizes:
    - 32x32
    - 64x64
    - 96x96
    - 128x128
- Brace yourself, this will take a while.

### Webapp

- The webapp is a proof of concept heavily inspired by [Zama's encrypted image filtering example found on hugging face](https://huggingface.co/spaces/zama-fhe/encrypted_image_filtering).
- To run the webapp locally the following command is needed:

```sh
make webapp
```

- A more detailed explanation can be found in [the webapps README file](/src/webapp/README.md).

## References

- [Introduction to homomorphic encryption](https://www.zama.ai/introduction-to-homomorphic-encryption)
- [Zama Whitepaper](https://whitepaper.zama.ai/).
- [Homomorphic Encryption 101](https://www.zama.ai/post/homomorphic-encryption-101).
- [Fully Homomorphic Encryption Using Ideal Lattices](https://www.cs.cmu.edu/~odonnell/hits09/gentry-homomorphic-encryption.pdf).
- [Concrete ML from Zama](https://docs.zama.ai/concrete-ml/).
- [Encrypted Image Filtering Using Homomorphic Encryption Blog Post](https://www.zama.ai/post/encrypted-image-filtering-using-homomorphic-encryption)
- [MNIST Concrete ML example](https://github.com/zama-ai/concrete-ml/tree/main/use_case_examples/mnist).
- [Convolutional Neural Network advanced example](https://github.com/zama-ai/concrete-ml/blob/main/docs/advanced_examples/ConvolutionalNeuralNetwork.ipynb).
- [`human-vs-non-human` dataset used for training](https://www.kaggle.com/code/aliasgartaksali/human-vs-non-human-binary-classification/input).

---
