# udacity-deep-learning

Implementation of the Project 2 of the "Machine Learning - Introduction Nanodegree by Udacity". The main task train a neural network, using PyTorch, to recognize flower names from pictures.

## Context

This project is part of the Machine Learning Nanodegree by [Udacity](https://www.udacity.com/), as a requeriment for the graduation.

## Construction

### Neural Network

The Neural Networking is built using [PyTorch](https://github.com/pytorch/pytorch), transfering learning from PyTorch's available models.

### Data Engineering

All the data treatment, initial training and testing of the netowrk was done using a Jupyter notebook, available in this repository.

## Usage

The result of this project is a command line utility that can train a network and later use it to perform predictions.

### network.py

Main file with the `Network()` class. It has all the necessary methods to train and load networks, including methods to save and load checkpoints of the training.

### train.py

Command line utility to train a neural network. Basic usage:

`$ train.py <data_dir>`: starts training using data available in the `data_dir` path. Expects `data_dir` to have images separated in `train`, `valid` and `test` folders.

`$ train.py <data_dir> --save_dir <path>`: indicates a directory to save checpoints during the training. Saves on current directory by default.

`$ train.py <data_dir> --arch <pytorch_model>`: sets the PyTorch model to transfer learnings from. Available architectures are `vgg16`, `resnet50` and `densenet161`. The default is `densenet161`.

`$ train.py <data_dir> --learning_rate <float>`: sets learning rate for the training. Default is `0.001`.

`$ train.py <data_dir> --hidden_units <float>`: sets size of hidden layer in the model's classifier. Default is `512`.

`$ trian.py <data_dir> --dropout <float>`: sets dropout rate to use during training. Default is `0.35`.

`$ trian.py <data_dir> --epochs <int>`: set number of training epochs. Default is `10`.

`$ trian.py <data_dir> --gpu`: enables GPU acceleration via CUDA, if available. It is _disabled_ by default.

### predict.py

Commnad line utility to predict a flower name. Basic usage:

`$ predict.py <img_path> <checkpoint_path>`: predict flower name from `<img_path>` using a previous trained network saved in `<checkpoint_path>`.

`$ predict.py <img_path> <checkpoint_path> --top_k <int>`: sets the number of classes (flower names) to return, with their probabilities. Default is `1`.

`$ predict.py <img_path> <checkpoint_path> --category_names <json_path>`: sets a path to a mapping of categories to real names. Expects a JSON file. By default, it uses no mapping, meaning that the utility will display the category number, and not the flower name.

`$ predict.py <img_path> <checkpoint_path> --gpu`: enables GPU acceleration via CUDA, if available. It is _disabled_ by default.