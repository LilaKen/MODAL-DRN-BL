import torch
import torch.nn as nn
import torch.nn.functional as F


class SCSNet(nn.Module):
    def __init__(self):
        super(SCSNet, self).__init__()
        # Define convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        # Define fully connected layers
        self.fc1 = nn.Linear(64 * 25 * 4, 1024)
        self.fc2 = nn.Linear(1024, 30)
        self.fc3 = nn.Linear(46, 1024)
        self.fc4 = nn.Linear(1024, 6 * 8 * 64)
        # Define more convolutional layers for the decoder
        self.conv3 = nn.Conv2d(6, 32, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(32, 16, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(16, 1, kernel_size=3, padding=1)
        # Flatten layer
        self.flatten = nn.Flatten()
        # Final fully connected layer
        self.linear = nn.Linear(8192, 10)  # Output feature dimension is 10

    def forward(self, x_node, x_load):
        # Forward propagation
        x = F.relu(self.conv1(x_node))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(-1, 64 * 25 * 4)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, 0.5)
        x = F.softplus(self.fc2(x))

        x_load = x_load.view(-1, 1 * 1 * 16)
        dna_full = torch.cat((x, x_load), dim=1)

        x = F.softplus(self.fc3(dna_full))
        x = F.relu(self.fc4(x))
        x = x.view(-1, 6, 8, 64)
        x = F.upsample(x, scale_factor=2, mode='nearest')
        x = F.relu(self.conv3(x))
        x = F.upsample(x, scale_factor=2, mode='nearest')
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = self.flatten(x)  # Flatten the data
        x = self.linear(x)  # Apply the fully connected layer
        return x
