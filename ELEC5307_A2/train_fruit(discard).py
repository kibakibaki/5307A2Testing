# train_fruit.py
"""
Train pretrained models (ResNet18, VGG16, VGG19, Simple CNN baseline)
on the fruit dataset located in ./train and plot both training and validation curves.

All models use ImageNet pretrained weights for fair comparison.
"""

import os, sys, time, argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt

# -----------------------------
# Argument Parser
# -----------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="resnet18",
                    choices=["resnet18", "vgg16", "vgg19", "cifarcnn"],
                    help="Choose model: resnet18, vgg16, vgg19, cifarcnn")
args = parser.parse_args()

# -----------------------------
# Configuration
# -----------------------------
train_dir = "./train"
num_epochs = 20
batch_size = 16
learning_rate = 1e-3
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\n============================")
print(f" Using device: {device}")
print(f" Model: {args.model}")
print(f"============================\n")

# -----------------------------
# Transformations 
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # fixed size
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ===========================================================
# ✅ MAIN SECTION
# ===========================================================
if __name__ == "__main__":
    torch.multiprocessing.freeze_support()

    # -----------------------------
    # Dataset & Split
    # -----------------------------
    full_dataset = datasets.ImageFolder(train_dir, transform=transform)
    num_classes = len(full_dataset.classes)
    total_size = len(full_dataset)
    val_size = int(0.2 * total_size)
    train_size = total_size - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    print(f"Found {train_size} training and {val_size} validation images across {num_classes} classes.\n")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    # -----------------------------
    # Model selection (all pretrained on ImageNet)
    # -----------------------------
    if args.model == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    elif args.model == "vgg16":
        model = models.vgg16_bn(weights=models.VGG16_BN_Weights.IMAGENET1K_V1)
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)

    elif args.model == "vgg19":
        model = models.vgg19(weights=models.models.VGG19_BN_Weights.IMAGENET1K_V1)
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)

    elif args.model == "cifarcnn":
        class SimpleCNN(nn.Module):
            def __init__(self, num_classes):
                super(SimpleCNN, self).__init__()
                self.conv_layers = nn.Sequential(
                    nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(),
                    nn.MaxPool2d(2),
                    nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
                    nn.MaxPool2d(2)
                )
                self.fc_layers = nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(64 * 56 * 56, 256), nn.ReLU(),
                    nn.Linear(256, num_classes)
                )

            def forward(self, x):
                x = self.conv_layers(x)
                x = self.fc_layers(x)
                return x

        model = SimpleCNN(num_classes)

    model = model.to(device)

    # -----------------------------
    # Loss, Optimizer, Scheduler
    # -----------------------------
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)

    # -----------------------------
    # Training Loop
    # -----------------------------
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    for epoch in range(1, num_epochs + 1):
        start = time.time()
        model.train()
        running_loss, correct, total = 0.0, 0, 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        epoch_train_loss = running_loss / total
        epoch_train_acc = correct / total

        # Validation
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        epoch_val_loss = val_loss / val_total
        epoch_val_acc = val_correct / val_total

        # Record
        train_losses.append(epoch_train_loss)
        val_losses.append(epoch_val_loss)
        train_accs.append(epoch_train_acc)
        val_accs.append(epoch_val_acc)

        scheduler.step(epoch_val_acc)

        print(f"Epoch [{epoch}/{num_epochs}] "
              f"Train Loss: {epoch_train_loss:.4f} | Val Loss: {epoch_val_loss:.4f} | "
              f"Train Acc: {epoch_train_acc*100:.2f}% | Val Acc: {epoch_val_acc*100:.2f}% | "
              f"Time: {time.time()-start:.1f}s")

    # -----------------------------
    # Save model
    # -----------------------------
    save_name = f"fruit_{args.model}_pretrained.pth"
    torch.save(model.state_dict(), save_name)
    print(f"\n✅ Training complete! Model saved to {save_name}")

    # -----------------------------
    # Plot Loss and Accuracy
    # -----------------------------
    epochs = range(1, num_epochs + 1)
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'b-o', label="Train Loss")
    plt.plot(epochs, val_losses, 'r-o', label="Val Loss")
    plt.title("Loss per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accs, 'g-o', label="Train Acc")
    plt.plot(epochs, val_accs, 'orange', marker='o', label="Val Acc")
    plt.title("Accuracy per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.tight_layout()
    plt.savefig(f"training_plot_{args.model}_pretrained.png")
    plt.show()
    sys.exit(0)
