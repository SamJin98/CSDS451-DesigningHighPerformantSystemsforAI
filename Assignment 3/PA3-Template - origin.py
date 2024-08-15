import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn as nn
import torch.optim as optim

# You should implement these for CIFAR-10. HINT: The dataset may be accessed with Torchvision.
training_data =
training_loader = 

# The cnn model class.
class EX_CNN(nn.Module):
    # Specify the structure of your model.
    def __init__(self, **kwargs):
        super(EX_CNN, self).__init__()
        self.convolution_layers = nn.Sequential(
            nn.Conv2d(in_channels = 1, out_channels = 32, 
                        kernel_size = (3, 3), stride = 1), 
            nn.ReLU(), 
            nn.MaxPool2d(kernel_size = (2, 2), stride = 2), 
            nn.BatchNorm2d(num_features = 32), 

            nn.Conv2d(in_channels = 32, out_channels = 32, 
                        kernel_size = (3, 3), stride = 1), 
            nn.ReLU(), 
            nn.MaxPool2d(kernel_size = (2, 2), stride = 2), 
            nn.BatchNorm2d(num_features = 32), 

            nn.Conv2d(in_channels = 32, out_channels = 32, 
                        kernel_size = (3, 3), stride = 1), 
            nn.ReLU(), 
            nn.MaxPool2d(kernel_size = (2, 2), stride = 2), 
            nn.BatchNorm2d(num_features = 32), 

            nn.Conv2d(in_channels = 32, out_channels = 32, 
                        kernel_size = (3, 3), stride = 1), 
            nn.ReLU(), 
            nn.MaxPool2d(kernel_size = (2, 2), stride = 2), 
            nn.BatchNorm2d(num_features = 32), 

            nn.Conv2d(in_channels = 32, out_channels = 32, 
                        kernel_size = (3, 3), stride = 1), 
            nn.ReLU(), 
            nn.MaxPool2d(kernel_size = (2, 2), stride = 2), 
            nn.BatchNorm2d(num_features = 32), 

            nn.Conv2d(in_channels = 32, out_channels = 32, 
                        kernel_size = (3, 3), stride = 1), 
            nn.ReLU(), 
            nn.MaxPool2d(kernel_size = (2, 2), stride = 2), 
            nn.BatchNorm2d(num_features = 32)
        )

        self.linear_layers = nn.Sequential(
            nn.Linear(in_features = 3200, out_features = 64), 
            nn.ReLU(), 
            nn.BatchNorm1d(num_features = 64), 

            nn.Linear(in_features = 64, out_features = 64), 
            nn.ReLU(), 
            nn.BatchNorm1d(num_features = 64), 

            nn.Linear(in_features = 64, out_features = 64), 
            nn.ReLU(), 
            nn.BatchNorm1d(num_features = 64), 

            nn.Linear(in_features = 64, out_features = 64), 
            nn.ReLU(), 
            nn.BatchNorm1d(num_features = 64),
            
            nn.Linear(in_features = 64, out_features = 9), 
            nn.Sigmoid()
        )
    
    # Define the data flow through your task graph.
    def forward(self, x):
        x = self.convolution_layers(x)
        x = torch.flatten(input = x, start_dim = 1)
        x = self.linear_layers(x)
        return x

# Instantiate your model.
Ex_Net = EX_CNN()

# If you want, you can print the structure of your model.
print(Ex_Net)

# Specify the loss function and optimizer that you will use.
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(params = Ex_Net.parameters(), lr = 0.001)

# Use an Nvidia GPU if available, otherwise use the CPU.
device = ("cuda" if torch.cuda.is_available() else "cpu")

# You can print out the device that you are using to verify that it is a GPU.
print("The device that you are using is: ", device)

# Send the model to the GPU.
Ex_Net.to(device = device)

# Run your model for "x" epochs.
for epoch in range(x):
    running_loss = 0.0

    # This iterates over the data within an epoch.
    for i, data in enumerate(iterable = training_loader, start = 0):
        inputs, labels = data['image'].to(device), data['label'].to(device)

        optimizer.zero_grad()

        outputs = Ex_Net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()

        # This prints the loss every 20 batches.
        if i % 20 == 19:
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 20:.3f}')
            running_loss = 0.0

# This is just here to let you know that your model has finished training.
print("Finished training.")