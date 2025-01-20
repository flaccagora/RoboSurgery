import torch
import torch.nn as nn
import torch.nn.functional as F

class ConditionalVAE(nn.Module):
    def __init__(self, latent_dim=128, condition_dim=4):
        super(ConditionalVAE, self).__init__()
        
        # Further reduced number of filters in encoder (approximately 1/4)
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        
        # Reduced flatten size
        self.flatten_size = 128 * 16 * 32
        
        # Further simplified condition encoder
        self.condition_encoder = nn.Sequential(
            nn.Linear(condition_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU()
        )
        
        # Reduced dimensions for mu and logvar layers
        self.fc_mu = nn.Linear(self.flatten_size + 256, latent_dim)
        self.fc_logvar = nn.Linear(self.flatten_size + 256, latent_dim)
        
        # Decoder input with reduced dimensions
        self.decoder_input = nn.Linear(latent_dim + condition_dim, self.flatten_size)
        
        # Decoder with further reduced number of filters
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.ConvTranspose2d(8, 3, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )
    
    def encode(self, x, c):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        c = self.condition_encoder(c)
        x = torch.cat([x, c], dim=1)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z, c):
        z = torch.cat([z, c], dim=1)
        x = self.decoder_input(z)
        x = x.view(x.size(0), 128, 16, 32)
        x = self.decoder(x)
        return x
    
    def forward(self, x, c):
        mu, logvar = self.encode(x, c)
        z = self.reparameterize(mu, logvar)
        return self.decode(z, c), mu, logvar

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def vae_loss(recon_x, x, mu, logvar, beta=1.0):
    """
    Calculate the VAE loss with KL divergence
    """
    # Reconstruction loss (binary cross entropy)
    # BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
    MSE = F.mse_loss(recon_x, x, reduction='sum')

    # KL divergence
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    beta_warmup = min(1.0, epoch / 10) * beta 

    return MSE + beta_warmup * KLD, MSE, KLD

# Hyperparameters
batch_size = 8
epochs = 2
learning_rate = 1e-4
limit = -1
save_interval = 2
wandb = False
save_dir = 'checkpoints'
import os
HERE = os.path.dirname(os.path.abspath(__file__))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    print(f"Training on {torch.cuda.get_device_name(0)}")
    # clear memory
    torch.cuda.empty_cache()
    torch.cuda.reset_accumulated_memory_stats()    

if wandb:
    import wandb
    wandb.init(project="conditional-vae")


import numpy as np
images = np.load(HERE + '/dataset/images.npy')
actions = np.load(HERE + '/dataset/actions.npy')

from torch.utils.data import Dataset, DataLoader
class CustomDataset(Dataset):
    def __init__(self, images, actions, transforms=None):
        """
        Args:
            data_list (list): List of dictionaries containing 'o', 'theta', 'qpos', and 'qpos_new'.
        """
        assert len(images) == len(actions), "Number of images and actions must match."

        self.observations = images
        self.latent = actions
        self.transforms = transforms

    def __len__(self):
        """Returns the total number of samples."""
        return len(self.observations)

    def __getitem__(self, idx):
        """
        Retrieve one sample of data by index.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            A dictionary with inputs and expected outputs as tensors.
        """
        # Extract the dictionary for the given index
        image = self.observations[idx]
        action = self.latent[idx]
        
        # Convert data to PyTorch tensors
        theta = torch.tensor(action, dtype=torch.float32)
        o = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1) / 255.0
        # pos = è costante in questo caso      

        # Inputs: qpos_new, o, theta
        inputs = {
            'theta': theta,
            'deform_obs': o    
        }
        
        # Output: qpos_new
        target = {
            'theta': theta,
            'deform_obs': o    
        }
        
        return inputs, target

# Instantiate the dataset
custom_dataset = CustomDataset(images[:limit], actions[:limit])

# Create a DataLoader
train_loader = DataLoader(custom_dataset, batch_size, shuffle=True)

# Model, optimizer, and loss
model = ConditionalVAE()
model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

from tqdm import tqdm
print("------TRAINING STARTED-----")
print(f"Model has {count_parameters(model):,} trainable parameters")
print(f"Training on {device}")
print(f"len Dataset: {len(custom_dataset)}")

for epoch in range(epochs):
    total_loss = 0
    total_recon = 0
    total_kl = 0
    
    # Progress bar
    pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}')
    
    for batch_idx, (data,_) in enumerate(pbar):
        # Move data to device
        condition = data['theta'].to(device)
        obs = data['deform_obs'].to(device)
        
        
        # Reset gradients
        optimizer.zero_grad()
        
        # Forward pass
        recon_batch, mu, logvar = model(obs, condition)
        
        # Compute loss
        loss, recon, kl = vae_loss(recon_batch, obs, mu, logvar)
        
        # Backward pass
        loss.backward()
        
        # Update parameters
        optimizer.step()
        
        # Update metrics
        total_loss += loss.item()
        total_recon += recon.item()
        total_kl += kl.item()
        
        # Update progress bar
        pbar.set_postfix({
            'loss': total_loss / (batch_idx + 1),
            'recon': total_recon / (batch_idx + 1),
            'kl': total_kl / (batch_idx + 1)
        })
    
    # Save model and generate samples at intervals
    if (epoch + 1) % save_interval == 0:
        # Save model checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': total_loss,
        }, f'{save_dir}/checkpoint_epoch_{epoch+1}.pt')
        
        # Generate and save sample reconstructions
        with torch.no_grad():
            # Get a batch of data
            sample_data, _  = next(iter(train_loader))
            obs = sample_data['deform_obs'].to(device)
            sample_condition = sample_data['theta'].to(device)
            
            # Generate reconstructions
            recon_data, _, _ = model(obs, sample_condition)
            
            # Save original and reconstructed images
            comparison = torch.cat([obs[:8], recon_data[:8]])
            # save_image(comparison.cpu(),
            #             f'{save_dir}/reconstruction_epoch_{epoch+1}.png',
            #             nrow=8)
            
            # Generate random samples
            random_z = torch.randn(8, model.fc_mu.out_features).to(device)
            random_samples = model.decode(random_z, sample_condition[:8])
            # save_image(random_samples.cpu(),
            #             f'{save_dir}/samples_epoch_{epoch+1}.png',
            #             nrow=8)
    
    # Print epoch summary
    print(f'Epoch {epoch+1}/{epochs}:')
    print(f'Average Loss: {total_loss / len(train_loader):.4f}')
    print(f'Average Reconstruction Loss: {total_recon / len(train_loader):.4f}')
    print(f'Average KL Divergence: {total_kl / len(train_loader):.4f}')
