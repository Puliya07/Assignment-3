import torch
import torch.nn as nn
import torch.nn.functional as F

class PestCNN(nn.Module):
    """
    Implements the 3-layer convolutional neural network 
    for 17-class pest classification.
    
    Architecture:
    Input -> [Conv(32) -> ReLU -> MaxPool] 
          -> [Conv(64) -> ReLU -> MaxPool]
          -> [Conv(128) -> ReLU -> MaxPool]
          -> Flatten
          -> [FC(128) -> ReLU -> Dropout(0.5)]
          -> Output(17)
    """
    def __init__(self):
        super(PestCNN, self).__init__()

        # Block 1: Input [3, 224, 224] -> Output [32, 112, 112]
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)

        # Block 2: Input [32, 112, 112] -> Output [64, 56, 56]
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)

        # Block 3: Input [64, 56, 56] -> Output [128, 28, 28]
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)

        # Shared 2x2 MaxPool layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Input  features: 128 filters * 28px height * 28px width = 100352
        self.fc1 = nn.Linear(in_features=128 * 28 * 28, out_features=128)

        # Dropout layer
        self.dropout = nn.Dropout(p=0.5)

        # Output layer: 128 -> 17 classes
        self.output = nn.Linear(in_features=128, out_features=17)

    def forward(self, x):

        x = self.pool(F.relu(self.conv1(x)))  # Block 1
        x = self.pool(F.relu(self.conv2(x)))  # Block 2
        x = self.pool(F.relu(self.conv3(x)))  # Block 3
        x = torch.flatten(x, start_dim=1)    # Flatten
        x = F.relu(self.fc1(x))               # Fully Connected
        x = self.dropout(x)                   # Dropout
        x = self.output(x)                    # Output Layer
        return x
    