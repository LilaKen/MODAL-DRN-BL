import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        shortcut = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += shortcut
        return x

class Stressnet(nn.Module):
    def __init__(self):
        super(Stressnet, self).__init__()
        # Define convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=9, padding=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        # Define residual blocks
        self.res_blocks = nn.Sequential(*[ResidualBlock(128) for _ in range(5)])
        # Define transposed convolutional layers for upsampling
        self.deconv1 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv2 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv4 = nn.Conv2d(32, 1, kernel_size=9, padding=4)
        # Flatten layer
        self.flatten = nn.Flatten()
        # Final fully connected layer
        self.linear = nn.Linear(1 * 108 * 16, 10)  # Output feature dimension is 10

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.res_blocks(x)
        x = F.relu(self.deconv1(x))
        x = F.relu(self.deconv2(x))
        x = self.conv4(x)
        x = self.flatten(x)  # Flatten the data
        x = self.linear(x)   # Apply the fully connected layer
        return x
#
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import Dataset, DataLoader
# import numpy as np
#
# class CustomDataset(Dataset):
#     def __init__(self, data, labels):
#         self.data = data
#         self.labels = labels
#
#     def __len__(self):
#         return len(self.data)
#
#     def __getitem__(self, idx):
#         sample = self.data[idx]
#         label = self.labels[idx]
#         return sample, label
#
# # Generate data
# train_data = np.random.randn(61834, 108, 16).astype(np.float32)
# train_labels = np.random.randn(61834, 10).astype(np.float32)  # Regression labels
#
# val_data = np.random.randn(20612, 108, 16).astype(np.float32)
# val_labels = np.random.randn(20612, 10).astype(np.float32)
#
# # Create datasets
# train_dataset = CustomDataset(train_data, train_labels)
# val_dataset = CustomDataset(val_data, val_labels)
#
# # DataLoader
# batch_size = 64
# train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
# val_loader = DataLoader(val_dataset, batch_size=batch_size)
#
# # Model, loss function and optimizer
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = Stressnet().to(device)
# criterion = nn.MSELoss()
# optimizer = optim.Adam(model.parameters(), lr=0.001)
#
# # Training process
# num_epochs = 10
# for epoch in range(num_epochs):
#     model.train()
#     running_loss = 0.0
#     for data, labels in train_loader:
#         data, labels = data.to(device), labels.to(device)
#         data = data.view(data.size(0), 1, 108, 16)
#         optimizer.zero_grad()
#         outputs = model(data)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()
#
#         running_loss += loss.item()
#
#     print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader)}")
#
# # Testing process
# model.eval()
# with torch.no_grad():
#     total_loss = 0
#     for data, labels in val_loader:
#         data, labels = data.to(device), labels.to(device)
#         data = data.view(data.size(0), 1, 108, 16)
#         outputs = model(data)
#         loss = criterion(outputs, labels)
#         total_loss += loss.item()
#
#     print(f"Validation Loss: {total_loss/len(val_loader)}")
