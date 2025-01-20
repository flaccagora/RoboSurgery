import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from torchvision.utils import make_grid

wandb_log = True

class ConditionalVAE(nn.Module):
    # ... [previous ConditionalVAE class implementation remains the same]
    pass

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def vae_loss(recon_x, x, mu, logvar, beta=1.0):
    MSE = F.mse_loss(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    beta_warmup = min(1.0, epoch / 10) * beta 
    return MSE + beta_warmup * KLD, MSE, KLD

# Hyperparameters
config = {
    "batch_size": 8,
    "epochs": 100,
    "learning_rate": 1e-4,
    "limit": -1,
    "save_interval": 2,
    "latent_dim": 128,
    "condition_dim": 4,
    "beta": 1.0
}

save_dir = 'checkpoints_wandb'
import os
os.makedirs(save_dir, exist_ok=True)
HERE = os.path.dirname(os.path.abspath(__file__))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    print(f"Training on {torch.cuda.get_device_name(0)}")
    torch.cuda.empty_cache()
    torch.cuda.reset_accumulated_memory_stats()    

# Initialize wandb
if wandb_log:
    wandb.init(
        project="conditional-vae",
        config=config,
        name=f"cvae_run_{wandb.util.generate_id()}"
    )

# Load and prepare data
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
custom_dataset = CustomDataset(images[:config['limit']], actions[:config['limit']])
train_loader = DataLoader(custom_dataset, config['batch_size'], shuffle=True)

# Model, optimizer, and loss
model = ConditionalVAE()
optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
model.to(device)

from tqdm import tqdm
print("------TRAINING STARTED-----")
print(f"Model has {count_parameters(model):,} trainable parameters")
print(f"Training on {device}")
print(f"len Dataset: {len(custom_dataset)}")

# Log model architecture
wandb.watch(model, log="all", log_freq=100)

def log_images(epoch, original, reconstructed, random_samples):
    """Helper function to log images to wandb"""
    # Create grids
    orig_grid = make_grid(original.cpu(), nrow=4, normalize=True)
    recon_grid = make_grid(reconstructed.cpu(), nrow=4, normalize=True)
    random_grid = make_grid(random_samples.cpu(), nrow=4, normalize=True)
    
    # Log to wandb
    wandb.log({
        "original_images": wandb.Image(orig_grid, caption=f"Original (Epoch {epoch})"),
        "reconstructed_images": wandb.Image(recon_grid, caption=f"Reconstructed (Epoch {epoch})"),
        "random_samples": wandb.Image(random_grid, caption=f"Random Samples (Epoch {epoch})")
    }, step=epoch)

for epoch in range(config['epochs']):
    model.train()
    total_loss = 0
    total_recon = 0
    total_kl = 0
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{config["epochs"]}')
    
    for batch_idx, (data, _) in enumerate(pbar):
        condition = data['theta'].to(device)
        obs = data['deform_obs'].to(device)
        
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(obs, condition)
        loss, recon, kl = vae_loss(recon_batch, obs, mu, logvar, config['beta'])
        
        loss.backward()
        optimizer.step()
        
        # Update metrics
        total_loss += loss.item()
        total_recon += recon.item()
        total_kl += kl.item()
        
        # Log batch metrics
        wandb.log({
            "batch_loss": loss.item(),
            "batch_recon_loss": recon.item(),
            "batch_kl_loss": kl.item(),
        }, step=epoch * len(train_loader) + batch_idx)
        
        pbar.set_postfix({
            'loss': total_loss / (batch_idx + 1),
            'recon': total_recon / (batch_idx + 1),
            'kl': total_kl / (batch_idx + 1)
        })
    
    # Calculate epoch metrics
    avg_loss = total_loss / len(train_loader)
    avg_recon = total_recon / len(train_loader)
    avg_kl = total_kl / len(train_loader)
    
    # Log epoch metrics
    wandb.log({
        "epoch": epoch,
        "avg_loss": avg_loss,
        "avg_recon_loss": avg_recon,
        "avg_kl_loss": avg_kl,
        "learning_rate": optimizer.param_groups[0]['lr']
    }, step=epoch)
    
    # Save model and generate samples at intervals
    if (epoch + 1) % config['save_interval'] == 0:
        # Save model checkpoint
        checkpoint_path = f'{save_dir}/checkpoint_epoch_{epoch+1}.pt'
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': total_loss,
        }, checkpoint_path)
        
        # Log model checkpoint to wandb
        wandb.save(checkpoint_path)
        
        # Generate and log sample images
        with torch.no_grad():
            # Get a batch of data
            sample_data, _ = next(iter(train_loader))
            obs = sample_data['deform_obs'].to(device)
            sample_condition = sample_data['theta'].to(device)
            
            # Generate reconstructions
            recon_data, _, _ = model(obs, sample_condition)
            
            # Generate random samples
            random_z = torch.randn(8, model.fc_mu.out_features).to(device)
            random_samples = model.decode(random_z, sample_condition[:8])
            
            # Log images to wandb
            log_images(epoch, obs[:8], recon_data[:8], random_samples)
    
    # Print epoch summary
    print(f'Epoch {epoch+1}/{config["epochs"]}:')
    print(f'Average Loss: {avg_loss:.4f}')
    print(f'Average Reconstruction Loss: {avg_recon:.4f}')
    print(f'Average KL Divergence: {avg_kl:.4f}')

# Finish the wandb run
wandb.finish()