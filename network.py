# python modules

import os
import json
from datetime import datetime
from PIL import Image

# torch imports

import torch
from torch import nn
from torch import optim
from torchvision import datasets, transforms, models

class Network:
    def __init__(self, device = "", arch = "", save_dir=""):
        if device:
            self.set_device(device)
        
        if arch:
            self.set_arch(arch)
        
        self.class_names = []
        self.set_save_dir(save_dir)
    
    def set_device(self, name):
        self.device = torch.device(name)
    
    def set_arch(self, name):
        if name == "vgg16":
            self.model = models.vgg16(pretrained=True)
            self.in_features = 25088
            
        elif name == "resnet50":
            self.model = models.resnet50(pretrained=True)
            self.in_features = 2048

        elif name == "densenet161":
            self.model = models.densenet161(pretrained=True)
            self.in_features = 2208
            
        else:
            exit("'{}' is not a supported pre-trained model. Please choose 'vgg16', 'resnet50' or 'densenet161'.".format(name))
        
        self.arch = name
    
    def set_save_dir(self, path):
        self.save_dir = path
        
    def set_data(self, data_dir):
        """
        Sets path to training data and applies transformations to this data
        Expects a folder structure separating train, validation and testing data
        """
        
        train_dir = os.path.join(data_dir, "train")
        valid_dir = os.path.join(data_dir, "valid")
        test_dir = os.path.join(data_dir, "test")

        size = 224
        
        data_transforms = {
            "train": transforms.Compose([
                transforms.RandomRotation(45),
                transforms.RandomHorizontalFlip(),
                transforms.RandomResizedCrop(size),
                transforms.ToTensor(),
                transforms.Normalize(
                    [0.485, 0.456, 0.406],
                    [0.229, 0.224, 0.225]
                )
            ]),
            
            "test": transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(size),
                transforms.ToTensor(),
                transforms.Normalize(
                    [0.485, 0.456, 0.406],
                    [0.229, 0.224, 0.225]
                )
            ]) # validation and testing will use the same transformations
        }

        image_datasets = {
            "train": datasets.ImageFolder(train_dir, transform=data_transforms["train"]),
            "valid": datasets.ImageFolder(valid_dir, transform=data_transforms["test"]),
            "test": datasets.ImageFolder(test_dir, transform=data_transforms["test"])
        }

        data_loaders = {
            "train": torch.utils.data.DataLoader(image_datasets["train"], batch_size=32, shuffle=True),
            "valid": torch.utils.data.DataLoader(image_datasets["valid"], batch_size=32, shuffle=True),
            "test": torch.utils.data.DataLoader(image_datasets["test"], batch_size=32, shuffle=True)
        }

        # self.class_to_idx = data_loaders["train"].dataset.class_to_idx
        self.data_loaders = data_loaders
        self.class_to_idx = image_datasets["train"].class_to_idx
    
    def set_class_names(self, json_path):
        with open(json_path, "r") as f:
            self.class_names = json.load(f)

    def train(self, hidden_units=512, dropout=0.35, learning_rate=0.001, epochs=10):
        """
        Trains the model
        Saves a checkpoint at every epoch
        """
        
        time_ = datetime.now().strftime("%Y-%m-%d %H:%M")
        
        self.epochs = epochs

        for param in self.model.parameters():
            param.requires_grad = False
            
        self.model.classifier = nn.Sequential(
            nn.Linear(self.in_features, int(self.in_features/2)),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(int(self.in_features/2), hidden_units),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_units, 102), # out features must be 102 as we have 102 categories
            nn.LogSoftmax(dim=1)
        )
        
        if self.arch == "resnet50":
            self.model.fc = self.model.classifier
            # fix seen here:
            # https://discuss.pytorch.org/t/element-0-of-tensors-does-not-require-grad-and-does-not-have-a-grad-fn/32908/3

        self.model.to(self.device)

        criterion = nn.NLLLoss()
        self.optimizer = optim.Adam(self.model.classifier.parameters(), lr=learning_rate)

        every = int(len(self.data_loaders["train"])/9)

        for e in range(epochs):
            self.current_epoch = e
            running_loss = 0
            
            print("\nepoch {}/{}:\n- - -".format(e+1, epochs))
            
            j = -1
            for images, labels in self.data_loaders["train"]:
                j += 1
                # print("{}/{}".format(j, len(self.data_loaders["train"])))
                
                self.model.train()
                
                images, labels = images.to(self.device), labels.to(self.device)
                
                self.optimizer.zero_grad()
                
                output = self.model.forward(images)
                loss = criterion(output, labels)
                loss.backward()
                
                self.optimizer.step()
                
                running_loss += loss.item()
                
                if j % every == 0:
                    test_loss = 0
                    accuracy = 0

                    self.model.eval()
                    
                    print("\nrunning validation...")

                    with torch.no_grad():
                        for images, labels in self.data_loaders["valid"]:
                            images, labels = images.to(self.device), labels.to(self.device)

                            output = self.model.forward(images)
                            test_loss += criterion(output, labels)

                            output_exp = torch.exp(output)
                            a, top_class = output_exp.topk(1, dim=1)
                            
                            correct = top_class == labels.view(*top_class.shape)
                            accuracy += torch.mean(correct.type(torch.FloatTensor))

                    print("training loss: {:.3f},".format(running_loss/len(self.data_loaders["train"])),
                          "test loss: {:.3f},".format(test_loss/len(self.data_loaders["valid"])),
                          "test accuracy: {:.3f}".format(accuracy/len(self.data_loaders["valid"])))
            
            self.save_checkpoint(time_ + " " + str(e+1) + ".pth")
            
    def save_checkpoint(self, name):
        save_info = {
            "epochs": {}
        }
        
        if hasattr(self, "arch"):
            save_info["arch"] = self.arch
            
        if hasattr(self, "model"):
            save_info["state_dict"] = self.model.state_dict()
            save_info["classifier"] = self.model.classifier
        
        if hasattr(self, "optimizer"):
            save_info["optimizer"] = self.optimizer
            save_info["optimizer_state_dict"] = self.optimizer.state_dict()
                    
        if hasattr(self, "class_to_idx"):
            save_info["class_to_idx"] = self.class_to_idx
                      
        if hasattr(self, "epochs"):
            save_info["epochs"]["total"] = self.epochs
            save_info["epochs"]["current"] = self.current_epoch

        file_path = os.path.join(self.save_dir, name)            
        torch.save(save_info, file_path)

    def load_checkpoint(self, name):
        location = "cpu"
        
        if torch.cuda.is_available():
            location = lambda storage, loc: storage.cuda()
        
        file_path = os.path.join(self.save_dir, name)
        checkpoint = torch.load(file_path, map_location = location)
        # "map_location" setting taken from here:
        # https://stackoverflow.com/questions/55759311/runtimeerror-cuda-runtime-error-35-cuda-driver-version-is-insufficient-for
        
        if "arch" in checkpoint.keys():
            self.set_arch(checkpoint["arch"])
    
            for param in self.model.parameters():
                param.requires_grad = False
        
        if "classifier" in checkpoint.keys():
            self.model.classifier = checkpoint["classifier"]
            self.model.load_state_dict(checkpoint["state_dict"])
            
        if "optimizer" in checkpoint.keys():
            self.optimizer = checkpoint["optimizer"]
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
        if "class_to_idx" in checkpoint.keys():
            self.class_to_idx = checkpoint["class_to_idx"]
            
        if "total" in checkpoint["epochs"].keys():
            self.epochs = checkpoint["epochs"]["total"]
            self.current_epoch = checkpoint["epochs"]["current"]
    
    def process_image(self, image_path):
        """
        Provides tranformations for images that will be used for predictions
        """
        
        data = Image.open(image_path)
        
        conv = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225]
            )
        ])
        
        return conv(data)    
    
    def predict(self, image_path, k=1):
        self.model.to(self.device)
        
        self.model.eval()
        
        image = self.process_image(image_path)   
        image = image.unsqueeze(0)
        # as seen here
        # https://discuss.pytorch.org/t/expected-stride-to-be-a-single-integer-value-or-a-list/17612/4
        
        image = image.to(self.device)
       
        with torch.no_grad():
            output = self.model.forward(image)
            
        result = output.topk(k, dim=1)
        
        probs = torch.exp(result[0])
        probs = probs.tolist()[0]
                    
        ids = [(value, key) for key, value in self.class_to_idx.items()]
        ids = dict(ids)
        
        classes = result[1]
        classes = classes.tolist()[0]   
        classes = [ids[x] for x in classes]
        
        if self.class_names:
            classes = [self.class_names[str(num)] for num in classes]
            
        self.model.train()
        
        return probs, classes