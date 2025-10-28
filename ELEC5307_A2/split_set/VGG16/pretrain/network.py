# network_vgg16.py
import torch
import torch.nn as nn
import os

class Network(nn.Module):
    def __init__(self, num_classes=1000, use_pretrained=True, weight_path="vgg16_bn-6c64b313.pth"):
        super(Network, self).__init__()

        # --------------------------
        # Define feature extractor
        # --------------------------
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Block 2
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Block 3
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Block 4
            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Block 5
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )

        # --------------------------
        # Classifier
        # --------------------------
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )

        # --------------------------
        # Load pretrained weights from local file
        # --------------------------
        if use_pretrained:
            if not os.path.exists(weight_path):
                raise FileNotFoundError(
                    f"❌ Cannot find pretrained weight file: {weight_path}\n"
                    "Please download 'vgg16_bn-6c64b313.pth' from:\n"
                    "https://download.pytorch.org/models/vgg16_bn-6c64b313.pth\n"
                    "and place it in the same directory as this script."
                )

            print(f"Loading pretrained weights from local file: {weight_path}")
            try:
                state_dict = torch.load(weight_path, map_location="cpu", weights_only=True)
            except TypeError:
                state_dict = torch.load(weight_path, map_location="cpu")

            self._load_pretrained_weights(state_dict)

    def _load_pretrained_weights(self, state_dict):
        # Remove final FC layer weights (ImageNet classifier)
        state_dict.pop("classifier.6.weight", None)
        state_dict.pop("classifier.6.bias", None)
        missing, unexpected = self.load_state_dict(state_dict, strict=False)
        print(f"✅ Loaded pretrained VGG16-BN weights (missing={missing}, unexpected={unexpected})")

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
