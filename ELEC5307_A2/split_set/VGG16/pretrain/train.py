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
import matplotlib.pyplot as plt

import torch
from torch import nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder

from network import Network # the network you used


# -----------------------------
# Log Setup: print to console AND save to file
# -----------------------------
class Logger(object):
    def __init__(self, filename=None):
        if filename is None:
            filename = f"train_log_{time.strftime('%Y%m%d_%H%M%S')}.txt"
        self.terminal = sys.stdout
        self.log = open(filename, "a", encoding="utf-8")
        print(f"\n📝 Logging to: {filename}\n")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()  

    def flush(self):
        pass

sys.stdout = Logger()


# =====================================
# Argument parsing
# =====================================
parser = argparse.ArgumentParser(description='script for training of project 2')
parser.add_argument('--cuda', action='store_true', default=False,
                    help='Use GPU if available.')
parser.add_argument('--epochs', type=int, default=20,
                    help='number of training epochs')
parser.add_argument('--batch_size', type=int, default=16,
                    help='batch size for training and validation')
parser.add_argument('--lr', type=float, default=1e-3,
                    help='initial learning rate')
args = parser.parse_args()

# =====================================
# Training process
# =====================================
def train_net(net, trainloader, valloader):
    device = torch.device('cuda' if args.cuda and torch.cuda.is_available() else 'cpu')
    net.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max',
                                                           factor=0.5, patience=3)

    best_val_acc = 0.0
    save_path = './project2.pth'
    num_epochs = args.epochs

    train_losses, val_losses, train_accuracies, val_accuracies = [], [], [], []

    for epoch in range(1, num_epochs + 1):
        net.train()
        running_loss, running_corrects, total = 0.0, 0, 0
        start_time = time.time()

        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            running_corrects += (preds == labels).sum().item()
            total += labels.size(0)

        train_loss = running_loss / total
        train_acc = running_corrects / total

        train_losses.append(train_loss)
        train_accuracies.append(train_acc)

        # ===== Validation =====
        net.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for inputs, labels in valloader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_loss /= val_total
        val_acc = val_correct / val_total

        val_losses.append(val_loss)
        val_accuracies.append(val_acc)

        scheduler.step(val_acc)

        print(f"Epoch [{epoch}/{num_epochs}]: "
              f"TrainLoss: {train_loss:.4f} | "
              f"TrainAcc: {train_acc:.4f} | "
              f"ValLoss: {val_loss:.4f} | "
              f"ValAcc: {val_acc:.4f} | "
              f"Time: {time.time() - start_time:.1f}s")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(net.state_dict(), save_path)
            print(f"Saved new best model to {save_path} (val_acc={best_val_acc:.4f})")

    print(f"Training complete. Best Val Accuracy: {best_val_acc:.4f}")

    # ===== Plotting =====
    epochs = range(1, num_epochs + 1)
    plt.figure(figsize=(10, 5))

    # --- Loss plot ---
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'b-o', label='Train Loss')
    plt.plot(epochs, val_losses, 'r-o', label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss per Epoch')
    plt.legend()

    # --- Accuracy plot ---
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracies, 'g-o', label='Train Acc')
    plt.plot(epochs, val_accuracies, 'orange', marker='o', label='Val Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy per Epoch')
    plt.legend()

    plt.tight_layout()
    plt.savefig("training_plot.png")
    plt.show()

    return best_val_acc, val_acc

##############################################

############################################
# Transformation definition
# NOTE:
# Write the train_transform here. We recommend you use
# Normalization, RandomCrop and any other transform you think is useful.

# train_transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(), 
#     transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
# ])

# val_transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(), 
#     transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
# ])



val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])

# -----------------------------
# Transformations (ImageNet standard)
# -----------------------------
train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.2, 0.2, 0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])

####################################

####################################
# Define the training dataset and dataloader.
# You can make some modifications, e.g. batch_size, adding other hyperparameters, etc.

train_image_path = '../../train/' 
validation_image_path = '../../val/' 

trainset = ImageFolder(train_image_path, train_transform)
valset = ImageFolder(validation_image_path, train_transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=16,
                                         shuffle=True, num_workers=0)
valloader = torch.utils.data.DataLoader(valset, batch_size=16,
                                         shuffle=False, num_workers=0)
####################################

# ==================================
# use cuda if called with '--cuda'.

num_classes = len(trainset.classes) 
print(f"Detected {num_classes} classes:", trainset.classes)

network = Network(num_classes=num_classes, use_pretrained=True)
if args.cuda:
    network = network.cuda()

# train and eval your trained network
# you have to define your own 
best_val_acc, final_val_acc = train_net(network, trainloader, valloader)

print("best validation accuracy:", best_val_acc)
print("final validation accuracy:", final_val_acc)

# ==================================
