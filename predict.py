# imports

import argparse
import torch
from network import Network

# args

cmd = argparse.ArgumentParser(description = "Analyzes an image of a flower and predicts its name. Uses a network previously trained on 'train.py' and saved as a checkpoint")

cmd.add_argument("img",
                 action="store",
                 help="The path to the image you want to take the predict on")

cmd.add_argument("checkpoint",
                 action="store",
                 help="The path to the model checkpoint you have previsouly trained")

cmd.add_argument("--top_k",
                 action="store",
                 dest="k",
                 default="1",
                 type=int,
                 help="Return top K most likely classes for the given image")

cmd.add_argument("--category_names",
                 action="store",
                 dest="cat_path",
                 help="Set a path to a mapping of categories to real names. Expects a JSON file")

cmd.add_argument("--gpu",
                 action="store_true",
                 dest="gpu",
                 help="Enable GPU usage, if avaiable")

args = cmd.parse_args()

# app

device = "cpu"

if args.gpu:
    if torch.cuda.is_available():
        device = "cuda"
    else:
        exit("CUDA is not available in your system. Please run without the --gpu flag")

model = Network()
model.set_device(device)
model.load_checkpoint(args.checkpoint)

if args.cat_path:
    model.set_class_names(args.cat_path)

probs, classes = model.predict(args.img, args.k)
output = zip(classes, probs)

i = 1
for name, prob in output:
    print("{}:".format(i),
          name.capitalize(),
          "({:.2f}%)".format(prob * 100))
    
    i += 1