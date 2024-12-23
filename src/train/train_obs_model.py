import pickle
# lod dataset
with open("dataset0.pkl", "rb") as f:
    dataset0 = pickle.load(f)

import torch
from torch.utils.data import Dataset, DataLoader

class CustomDataset(Dataset):
    def __init__(self, data_list):
        """
        Args:
            data_list (list): List of dictionaries containing 'o', 'theta', 'qpos', and 'qpos_new'.
        """
        self.data_list = data_list

    def __len__(self):
        """Returns the total number of samples."""
        return len(self.data_list)

    def __getitem__(self, idx):
        """
        Retrieve one sample of data by index.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            A dictionary with inputs and expected outputs as tensors.
        """
        # Extract the dictionary for the given index
        data = self.data_list[idx]
        
        # Convert data to PyTorch tensors
        theta = torch.tensor(data['theta'], dtype=torch.float32).flatten()
        o_new = torch.tensor(data['deform_obs'], dtype=torch.float32)
        pos = torch.tensor(data['pos'], dtype=torch.float32)      
        

        # Inputs: qpos_new, o, theta
        inputs = {
            'theta': theta,
            'pos': pos
        }
        
        # Output: qpos_new
        target = {
             "deform_obs": o_new        
             }
        
        return inputs, target


# Instantiate the dataset
custom_dataset = CustomDataset(dataset0)

# Create a DataLoader
data_loader = DataLoader(custom_dataset, batch_size=1024, shuffle=True)


import torch.nn as nn
import torch.nn.functional as F

class NN(nn.Module):
    def __init__(self):
        super(NN, self).__init__()

        self.f1 = nn.Linear(6, 128)
        self.f2 = nn.Linear(128, 128)
        self.f3 = nn.Linear(128, 128)
        self.f4 = nn.Linear(128, 1)
        
    def forward(self, pos,theta):
        x = torch.cat([pos,theta], dim=1)
        x = F.relu(self.f1(x))
        x = F.relu(self.f2(x))
        x = F.relu(self.f3(x))
        x = F.sigmoid(self.f4(x))
        return x


# train network 
import torch.optim as optim
from tqdm import tqdm
# Instantiate the model
model = NN()


# Define the loss function
criterion = nn.BCELoss()

# Define the optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Set the model in training mode
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
model.train()


# tqdm progress bar
pbar = tqdm(total=len(data_loader),desc="Training")
pbar.refresh()
pbar.reset()

# Iterate through the DataLoader
for epoch in range(25):
    running_loss = 0.0
    pbar.set_description(f"Epoch {epoch}")
    for i, (inputs, target) in enumerate(data_loader):
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(inputs['pos'].to(device).to(device),inputs['theta'].to(device))

        # Compute the loss
        loss = criterion(outputs, target['deform_obs'].to(device))

        # Backward pass
        running_loss += loss.item()
        loss.backward()

        # Update the weights
        optimizer.step()

        pbar.update(1)
    pbar.reset()
    print("runningLoss:", running_loss/len(data_loader))
    if (epoch + 1) % 5 == 0:
        print(f"Epoch {epoch + 1}")
        torch.save(model.state_dict(), f"obs_model_{epoch}.pth")

# save model
torch.save(model.state_dict(), "obs_model.pth")
print("Model saved")