import torch
import torch.nn as nn
import torch.nn.functional as F


class Inbetween(nn.Module):
    def __init__(self):
        super(Inbetween, self).__init__()
        # Define layers for the model
        # First layer: shape, load, boundary
        self.conv1_shape = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv1_load = nn.Conv2d(1, 8, kernel_size=3, padding=1)
        self.conv1_boundary = nn.Conv2d(1, 8, kernel_size=3, padding=1)
        # Second layer: shared
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv2_load = nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1)
        self.conv2_boundary = nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1)
        # Define fully connected layers
        self.fc1 = nn.Linear(64 * 13 * 2, 1024)
        self.fc1_load = nn.Linear(16 * 1 * 8, 128)
        self.fc1_boundary = nn.Linear(16 * 1 * 4, 128)
        self.fc2 = nn.Linear(1024, 32)
        self.fc2_load = nn.Linear(128, 8)
        self.fc2_boundary = nn.Linear(128, 8)
        # Feature fusion
        self.fc3 = nn.Linear(48, 1024)
        # Decoder part
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv3 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(32, 16, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(16, 1, kernel_size=3, padding=1)
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(1 * 8 * 32, 10)  # Output feature dimension is 10

    def forward(self, x_shape, x_load, x_boundary):
        # Forward propagation
        x_shape = F.relu(self.conv1_shape(x_shape))
        x_load = F.relu(self.conv1_load(x_load))
        x_boundary = F.relu(self.conv1_boundary(x_boundary))
        x_shape = F.max_pool2d(x_shape, 2)
        x_load = F.max_pool2d(x_load, 1)
        x_boundary = F.max_pool2d(x_boundary, 2)
        x_shape = F.relu(self.conv2(x_shape))
        x_load = F.relu(self.conv2_load(x_load))
        x_boundary = F.relu(self.conv2_boundary(x_boundary))
        x_shape = F.max_pool2d(x_shape, 2)
        x_load = F.max_pool2d(x_load, 1)
        x_boundary = F.max_pool2d(x_boundary, 1)
        # Flatten
        x_shape = x_shape.view(-1, 64 * 13 * 2)
        x_load = x_load.view(-1, 16 * 1 * 8)
        x_boundary = x_boundary.view(-1, 16 * 1 * 4)
        x_shape = F.relu(self.fc1(x_shape))
        x_load = F.relu(self.fc1_load(x_load))
        x_boundary = F.relu(self.fc1_boundary(x_boundary))
        x_shape = F.relu(self.fc2(x_shape))
        x_load = F.relu(self.fc2_load(x_load))
        x_boundary = F.relu(self.fc2_boundary(x_boundary))
        # Feature fusion
        x = torch.cat((x_shape, x_load, x_boundary), dim=1)
        x = F.relu(self.fc3(x))
        # Decode
        x = x.view(-1, 64, 2, 8)
        x = self.upsample(x)
        x = F.relu(self.conv3(x))
        x = self.upsample(x)
        x = F.relu(self.conv4(x))
        x = self.conv5(x)
        x = self.flatten(x)  # Flatten the data
        x = self.linear(x)  # Apply the fully connected layer
        return x


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np


class CustomDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.labels[idx]
        return sample, label


# Generate data
train_data = np.random.randn(61834, 108, 16).astype(np.float32)
train_labels = np.random.randn(61834, 10).astype(np.float32)  # Regression labels

val_data = np.random.randn(20612, 108, 16).astype(np.float32)
val_labels = np.random.randn(20612, 10).astype(np.float32)

# Create datasets
train_dataset = CustomDataset(train_data, train_labels)
val_dataset = CustomDataset(val_data, val_labels)

# DataLoader
batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

# Model, loss function and optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Inbetween().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training process
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for data, labels in train_loader:
        data, labels = data.to(device), labels.to(device)
        data = data.view(data.size(0), 1, 108, 16)

        # Extract first 8 rows
        removed_rows = data[:, :, :8, :]
        remaining_data = data[:, :, 8:, :]

        # Keypoint rows
        keypoint_rows = removed_rows[:, :, :2, :]

        # Line rows
        line_rows = removed_rows[:, :, 2:3, :]

        # DOF rows
        dof_rows = removed_rows[:, :, 3:7, :]

        # Load rows
        load_rows = removed_rows[:, :, 7:8, :]

        remaining_data = torch.cat([keypoint_rows, line_rows, remaining_data], dim=2)
        x_shape = remaining_data
        x_load = load_rows
        x_boundary = dof_rows
        optimizer.zero_grad()
        outputs = model(x_shape, x_load, x_boundary)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(train_loader)}")

# Testing process
model.eval()
with torch.no_grad():
    total_loss = 0
    for data, labels in val_loader:
        data, labels = data.to(device), labels.to(device)
        data = data.view(data.size(0), 1, 108, 16)

        # Extract first 8 rows
        removed_rows = data[:, :, :8, :]
        remaining_data = data[:, :, 8:, :]

        # Keypoint rows
        keypoint_rows = removed_rows[:, :, :2, :]

        # Line rows
        line_rows = removed_rows[:, :, 2:3, :]

        # DOF rows
        dof_rows = removed_rows[:, :, 3:7, :]

        # Load rows
        load_rows = removed_rows[:, :, 7:8, :]

        remaining_data = torch.cat([keypoint_rows, line_rows, remaining_data], dim=2)
        x_shape = remaining_data
        x_load = load_rows
        x_boundary = dof_rows
        outputs = model(x_shape, x_load, x_boundary)
        loss = criterion(outputs, labels)
        total_loss += loss.item()

    print(f"Validation Loss: {total_loss / len(val_loader)}")
