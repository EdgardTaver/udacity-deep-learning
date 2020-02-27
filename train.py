# imports

import argparse
import torch
from network import Network

# args

description = "Trains a neural network to predict flower names from images"
cmd = argparse.ArgumentParser(description = description)

cmd.add_argument("data_dir",
                 action="store",
                 help="Path to directory containing images for training")

cmd.add_argument("--save_dir",
                 action="store",
                 dest="save_dir",
                 default="checkpoints",
                 help="Path to directory in which traning checkpoints will be saved")

cmd.add_argument("--arch",
                 action="store",
                 dest="arch",
                 default="densenet161",
                 help="Set a pre-trained model to work with")

cmd.add_argument("--learning_rate",
                 action="store",
                 dest="learning_rate",
                 default=0.001,
                 type=float,
                 help="Set learning rate for the training")

cmd.add_argument("--hidden_units",
                 action="store",
                 dest="hidden_units",
                 default=512,
                 type=int,
                 help="Set number of units for the hidden layer used in the model's classifier")

cmd.add_argument("--dropout",
                 action="store",
                 dest="dropout",
                 default=0.35,
                 type=float,
                 help="Set dropout rate for the classifier")

cmd.add_argument("--epochs",
                 action="store",
                 dest="epochs",
                 default=10,
                 type=int,
                 help="Set number of training epochs")

cmd.add_argument("--gpu",
                 action="store_true",
                 dest="gpu",
                 help="Enable GPU usage, if avaiable")

args = cmd.parse_args()

# app

device = "cpu"
# default device is "cpu"

if args.gpu:
    if torch.cuda.is_available():
        device = "cuda"
    else:
        exit("CUDA is not available in your system. Please run without the --gpu flag")

model = Network()
model.set_device(device)
model.set_arch(args.arch)
model.set_data(args.data_dir)
model.set_save_dir(args.save_dir)

model.train(hidden_units=args.hidden_units,
            dropout=args.dropout,
            learning_rate=args.learning_rate,
            epochs=args.epochs)
