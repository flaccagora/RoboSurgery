import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from torchvision.utils import make_grid
import torch.optim.lr_scheduler as lr_scheduler

wandb_log = True
wandb_run_name = "cvae_run_k6kfrpkd"
continue_training = True
if continue_training == True and wandb_log == True:
    assert wandb_run_name is not None, "Please provide the name of the run to continue training or disable wandb logging"

class ConditionalVAE(nn.Module):
    def __init__(self, latent_dim=128, condition_dim=4, hidden_dim=256):
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
        
        # Prior network p(z|s)
        self.prior = nn.Sequential(
            nn.Linear(4, hidden_dim),  # Input is just the label
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.prior_mean = nn.Linear(hidden_dim, latent_dim)
        self.prior_logvar = nn.Linear(hidden_dim, latent_dim)

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

    def decode(self, z, c):
        z = torch.cat([z, c], dim=1)
        x = self.decoder_input(z)
        x = x.view(x.size(0), 128, 16, 32)
        x = self.decoder(x)
        return x
    
    def get_prior(self,c):
        c = self.prior(c)
        return self.prior_mean(c), self.prior_logvar(c)

    def forward(self, x, c):
        mu, logvar = self.encode(x, c)
        z = self.reparameterize(mu, logvar)
        return self.decode(z, c), mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def vae_loss(self, x, c, beta=1):
        
        # forward
        recon_x, mu, logvar = self.forward(x, c)

        # priors
        p_mu, p_logvar = self.get_prior(c)

        # Reconstruction loss
        reconstruction_loss = F.binary_cross_entropy(
            recon_x, x, reduction='none'
        ).sum(dim=[1,2,3]).mean()
         
        kl_loss = -0.5 * torch.sum(
            1 + logvar - p_logvar - 
            (mu - p_mu)**2/torch.exp(p_logvar) - 
            torch.exp(logvar)/torch.exp(p_logvar),
            dim=1
        ).mean()
        
        return reconstruction_loss + beta * kl_loss, reconstruction_loss, kl_loss

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Hyperparameters
config = {
    "batch_size": 64,
    "epochs": 200,
    "learning_rate": 1e-4,
    "limit": -1,
    "save_interval": 10,
    "latent_dim": 128,
    "condition_dim": 4,
    "warmup_epochs": 8,
    "default_beta": 1.0
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

warmup_scheduler = lr_scheduler.LambdaLR(
    optimizer,
    lr_lambda=lambda e: min(1.0, (e + 1) / config["warmup_epochs"])
)
scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.97)
model.to(device)

start_epoch = 0
if continue_training:
    import glob
    last_checkpoint = max(glob.glob(HERE / f'checkpoint_epoch_*.pt'), key=os.path.getctime)
    checkpoint = torch.load('checkpoint_epoch_120.ptrom')
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    start_epoch = checkpoint['epoch']
    print(f"Continuing training from epoch {start_epoch}")

# Initialize wandb
if wandb_log:
    wandb.init(
        project="conditional-vae",
        config=config,
        name=f"cvae_run_{wandb.util.generate_id()}" if wandb_run_name is None else wandb_run_name
    )


from tqdm import tqdm
print("------TRAINING STARTED-----")
print(f"Model has {count_parameters(model):,} trainable parameters")
print(f"Training on {device}")
print(f"len Dataset: {len(custom_dataset)}")

# Log model architecture
if wandb_log:
    wandb.watch(model, log="all", log_freq=100)

def log_images(epoch, original, reconstructed, random_samples):
    """Helper function to log images to wandb"""
    # Create grids
    orig_grid = make_grid(original.cpu(), nrow=4, normalize=True)
    recon_grid = make_grid(reconstructed.cpu(), nrow=4, normalize=True)
    random_grid = make_grid(random_samples.cpu(), nrow=4, normalize=True)
    
    # Log to wandb
    if wandb_log:
        wandb.log({
            "original_images": wandb.Image(orig_grid, caption=f"Original (Epoch {epoch})"),
            "reconstructed_images": wandb.Image(recon_grid, caption=f"Reconstructed (Epoch {epoch})"),
            "random_samples": wandb.Image(random_grid, caption=f"Random Samples (Epoch {epoch})")
        })

for epoch in range(start_epoch, config['epochs']):
    model.train()
    total_loss = 0
    total_recon = 0
    total_kl = 0
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{config["epochs"]}')
    
    for batch_idx, (data, _) in enumerate(pbar):
        condition = data['theta'].to(device)
        obs = data['deform_obs'].to(device)
        
        optimizer.zero_grad()
        # recon_batch, mu, logvar = model(obs, condition)
        # loss, recon, kl = vae_loss(recon_batch, obs, mu, logvar, config['beta'])
        beta_warmup = min(1.0, epoch / 15) * config['default_beta'] 

        loss, recon, kl = model.vae_loss(obs, condition, beta_warmup)

        loss.backward()
        optimizer.step()
        
        # Update metrics
        total_loss += loss.item()
        total_recon += recon.item()
        total_kl += kl.item()
        
        if wandb_log:
            # Log batch metrics
            wandb.log({
                "batch_loss": loss.item(),
                "batch_recon_loss": recon.item(),
                "batch_kl_loss": kl.item(),
            })
        
        pbar.set_postfix({
            'loss': total_loss / (batch_idx + 1),
            'recon': total_recon / (batch_idx + 1),
            'kl': total_kl / (batch_idx + 1)
        })
    
    if epoch < config["warmup_epochs"]:
        warmup_scheduler.step()
    else:
        scheduler.step()
    
    # Calculate epoch metrics
    avg_loss = total_loss / len(train_loader)
    avg_recon = total_recon / len(train_loader)
    avg_kl = total_kl / len(train_loader)
    
    if wandb_log:
        # Log epoch metrics
        wandb.log({
            "epoch": epoch,
            "avg_loss": avg_loss,
            "avg_recon_loss": avg_recon,
            "avg_kl_loss": avg_kl,
            "learning_rate": optimizer.param_groups[0]['lr'],
            "beta": beta_warmup,
        })
    
    # Save model and generate samples at intervals
    if (epoch + 1) % config['save_interval'] == 0:
        # Save model checkpoint
        checkpoint_path = f'{save_dir}/checkpoint_epoch_{epoch+1}.pt'
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'loss': total_loss,
        }, checkpoint_path)
        
        if wandb_log:
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
            if wandb_log:
                log_images(epoch, obs[:8], recon_data[:8], random_samples)
    
    # Print epoch summary
    print(f'Epoch {epoch+1}/{config["epochs"]}:')
    print(f'Average Loss: {avg_loss:.4f}')
    print(f'Average Reconstruction Loss: {avg_recon:.4f}')
    print(f'Average KL Divergence: {avg_kl:.4f}')

# Finish the wandb run
if wandb_log:
    wandb.finish()