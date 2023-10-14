# Human in picture using Concrete ML

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

### Python version

- It is recommended to use [Python 3.10.12](https://www.python.org/downloads/release/python-31012/).
- Or if you are using [pyenv](https://github.com/pyenv/pyenv)

```shell
pyenv install 3.10.12
```

## Installation

### Dependencies

- To install dependencies you will need to run `make deps`.

### Dataset

- To download the dataset you will need to run `make data`.

## Resources

- [MNIST Concrete ML example](https://github.com/zama-ai/concrete-ml/tree/main/use_case_examples/mnist).
- [Convolutional Neural Network advanced example](https://github.com/zama-ai/concrete-ml/blob/main/docs/advanced_examples/ConvolutionalNeuralNetwork.ipynb).
- [Dataset used for training](https://www.kaggle.com/code/aliasgartaksali/human-vs-non-human-binary-classification/input).

## Reading material

- [Zama Whitepaper](https://whitepaper.zama.ai/).
- [Homomorphic Encryption 101](https://www.zama.ai/post/homomorphic-encryption-101).
- [Fully Homomorphic Encryption Using Ideal Lattices](https://www.cs.cmu.edu/~odonnell/hits09/gentry-homomorphic-encryption.pdf).
- [Concrete ML from Zama](https://docs.zama.ai/concrete-ml/).
