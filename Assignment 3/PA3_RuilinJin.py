import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import time
import matplotlib.pyplot as plt

# CIFAR-10 transformations
transform = transforms.Compose([
  transforms.ToTensor(),
  transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# CIFAR-10 datasets and dataloaders
training_data = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
training_loader = DataLoader(training_data, batch_size=64, shuffle=True)

# The CNN model class
class EX_CNN(nn.Module):
  def __init__(self):
    super(EX_CNN, self).__init__()
    self.convolution_layers = nn.Sequential(
        nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.BatchNorm2d(32),

        nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.BatchNorm2d(64),

        nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.BatchNorm2d(128)
    )

    # Calculate the flattened feature size for the linear layer
    self._to_linear = None
    self._calculate_to_linear(32, 32)

    self.linear_layers = nn.Sequential(
        nn.Linear(in_features=self._to_linear, out_features=512),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.BatchNorm1d(512),

        nn.Linear(in_features=512, out_features=256),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.BatchNorm1d(256),

        nn.Linear(in_features=256, out_features=10) # CIFAR-10 has 10 classes
    )

  # Forward pass calculation for output size of last conv layer
  def _calculate_to_linear(self, l, w):
    x = torch.zeros(1, 3, l, w)
    self._to_linear = self.convolution_layers(x).view(-1).size(0)

  def forward(self, x):
    x = self.convolution_layers(x)
    x = x.view(-1, self._to_linear)  # Flatten the output for the linear layer
    x = self.linear_layers(x)
    return F.log_softmax(x, dim=1)

# PyTorch Lightning module
class LightningEX_CNN(pl.LightningModule):
  def __init__(self):
    super().__init__()
    self.model = EX_CNN()

  def forward(self, x):
    return self.model(x)

  def training_step(self, batch, batch_idx):
    inputs, labels = batch
    outputs = self(inputs)
    loss = nn.NLLLoss()(outputs, labels)
    self.log('train_loss', loss)
    return loss

  def configure_optimizers(self):
    return optim.Adam(self.parameters(), lr=0.001)

# List of batch sizes to test
batch_sizes = [32, 64, 128, 256]
training_times = []

# Loop over each batch size
for batch_size in batch_sizes:
  # Create the DataLoader with the current batch size
  training_loader = DataLoader(training_data, batch_size=batch_size, shuffle=True)

  # Instantiate the Lightning model for each batch size to reset weights
  lightning_model = LightningEX_CNN()

  # Initialize a trainer with the data parallel strategy and 1 GPU
  trainer = pl.Trainer(max_epochs=2, accelerator="gpu", devices=-1, strategy='auto')

  # Start timing
  start_time = time.time()

  # Train the model
  trainer.fit(lightning_model, training_loader)

  # End timing
  end_time = time.time()

  # Calculate and store the training time
  training_time = end_time - start_time
  training_times.append(training_time)

  print(f"Training time for batch size {batch_size}: {training_time:.2f} seconds")



# Plotting the training times
plt.figure(figsize=(10, 5))
plt.plot(batch_sizes, training_times, marker='o')
plt.title('Training Time vs Batch Size')
plt.xlabel('Batch Size')
plt.ylabel('Training Time (seconds)')
plt.xticks(batch_sizes)
plt.grid(True)
plt.show()
