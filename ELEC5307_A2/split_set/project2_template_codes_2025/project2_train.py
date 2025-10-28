'''
this script is for the training code of Project 2..

-------------------------------------------
INTRO:
You can change any parts of this code

-------------------------------------------
'''

# import the packages
import argparse
import logging
import sys
import time
import os

import torch
from torch import nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder

from network import Network # the network you used

parser = argparse.ArgumentParser(description= \
                                     'scipt for training of project 2')
parser.add_argument('--cuda', action='store_true', default=False,
                    help='Used when there are cuda installed.')
args = parser.parse_args()

# training process. 
def train_net(net, trainloader, valloader):
########## ToDo: Your codes goes below #######
    val_accuracy = 0
    # val_accuracy is the validation accuracy of each epoch. You can save your model base on the best validation accuracy.

    return val_accuracy

##############################################

############################################
# Transformation definition
# NOTE:
# Write the train_transform here. We recommend you use
# Normalization, RandomCrop and any other transform you think is useful.

train_transform = transforms.Compose([
    transforms.RandomCrop(224),
    transforms.ToTensor(), 
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

####################################

####################################
# Define the training dataset and dataloader.
# You can make some modifications, e.g. batch_size, adding other hyperparameters, etc.

train_image_path = '../train/' 
validation_image_path = '../validation/' 

trainset = ImageFolder(train_image_path, train_transform)
valset = ImageFolder(validation_image_path, train_transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                         shuffle=True, num_workers=2)
valloader = torch.utils.data.DataLoader(valset, batch_size=4,
                                         shuffle=True, num_workers=2)
####################################

# ==================================
# use cuda if called with '--cuda'.

network = Network()
if args.cuda:
    network = network.cuda()

# train and eval your trained network
# you have to define your own 
val_acc = train_net(network, trainloader, valloader)

print("final validation accuracy:", val_acc)

# ==================================
