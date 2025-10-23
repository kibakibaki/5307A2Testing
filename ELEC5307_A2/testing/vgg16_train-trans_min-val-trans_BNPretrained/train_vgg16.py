# train_vgg16.py
"""
Train a pretrained VGG16 (with BatchNorm) model on the fruit dataset located in ./train
and plot both training and validation curves.
"""

import os, sys, time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt

train_dir = "./train"
num_epochs = 20
batch_size = 16
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\n============================")
print(f" Using device: {device}")
print(f" Model: VGG16")
print(f"============================\n")

train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.2, 0.2, 0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])

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

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])

# ===========================================================
# ✅ MAIN SECTION
# ===========================================================
if __name__ == "__main__":
    torch.multiprocessing.freeze_support()

    g = torch.Generator().manual_seed(42)
    base_dataset = datasets.ImageFolder(train_dir)

    num_classes = len(base_dataset.classes)
    total_size = len(base_dataset)
    val_size = int(0.2 * total_size)
    train_size = total_size - val_size

    train_dataset, val_dataset = random_split(base_dataset, [train_size, val_size], generator=g)

    train_dataset.dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    val_dataset.dataset   = datasets.ImageFolder(train_dir, transform=val_transform)

    # DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

    model = models.vgg16_bn(weights=models.VGG16_BN_Weights.IMAGENET1K_V1)
    for p in model.features.parameters():
        p.requires_grad = False
    model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),
                                lr=1e-3, momentum=0.9, weight_decay=5e-4)

    train_losses, val_losses, train_accs, val_accs = [], [], [], []

    for epoch in range(1, num_epochs + 1):
        if epoch == 6:
            for p in model.features.parameters():
                p.requires_grad = True
            optimizer = torch.optim.SGD(model.parameters(), lr=1e-4, momentum=0.9, weight_decay=5e-4)

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
        train_losses.append(epoch_train_loss)
        val_losses.append(epoch_val_loss)
        train_accs.append(epoch_train_acc)
        val_accs.append(epoch_val_acc)

        print(f"Epoch [{epoch}/{num_epochs}] "
              f"Train Loss: {epoch_train_loss:.4f} | Val Loss: {epoch_val_loss:.4f} | "
              f"Train Acc: {epoch_train_acc*100:.2f}% | Val Acc: {epoch_val_acc*100:.2f}% | "
              f"Time: {time.time()-start:.1f}s")

    torch.save(model.state_dict(), "fruit_vgg16_pretrained.pth")
    print("\n✅ Training complete! Model saved to fruit_vgg16_pretrained.pth")

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
    plt.savefig("training_plot_vgg16.png")
    plt.show()
    sys.exit(0)
