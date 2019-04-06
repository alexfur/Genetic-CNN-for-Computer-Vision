# A Genetic Algorithm for Evolving CNN Hyper-Parameters

## What is this?

CNN hyper-parameters are difficult to tune manually. This project uses a Genetic Algorithm to automatically *evolve* optimal hyper-parameter values of a Convolutional Neural Network (CNN) for classifying clothing items in the Fashion-MNIST dataset.

## Tech

TensorFlow was used to create the CNN model and the Genetic Algorithm was written from scratch.

## Setting up and Running

Clone and cd into this git repository. Then, use pipenv to install dependencies and setup a virtual environment.

```
$ pipenv install
```

Now activate the project's virtualenv:

```
$ pipenv shell
```

And run main.py:

```
$ python main.py
```

For toggling parameters such as evolution and the intelligent survival function, modify config.yaml in the configs directory.
