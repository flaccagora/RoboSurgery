import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np

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
    
    def vae_loss(self, x, c, epoch, beta=1):
        
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
        
        beta_warmup = torch.min(1,epoch/10) * beta

        return reconstruction_loss + beta_warmup * kl_loss, reconstruction_loss, kl_loss

def train_vae_based(device='cuda' if torch.cuda.is_available() else 'cpu'):
    # Hyperparameters
    batch_size = 12
    n_epochs = 100
    lr = 1e-4

    # Load MNIST
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    
    train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    model = ConditionalVAE().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    train_losses = []
    reconstruction_losses = []
    kl_losses = []
    
    for epoch in range(n_epochs):
        model.train()
        epoch_losses = []
        epoch_rec_losses = []
        epoch_kl_losses = []
        
        for images, labels in tqdm(train_loader, desc=f'Epoch {epoch+1}/{n_epochs}'):
            images = images.to(device)
            states = labels.float().view(-1, 1).to(device)
            
            loss, reconstruction_losses, kl_loss = model.vae_loss(images, states, epoch)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_losses.append(loss.item())
            epoch_rec_losses.append(reconstruction_loss.item())
            epoch_kl_losses.append(kl_loss.item())
        
        avg_loss = np.mean(epoch_losses)
        avg_rec_loss = np.mean(epoch_rec_losses)
        avg_kl_loss = np.mean(epoch_kl_losses)
        
        train_losses.append(avg_loss)
        reconstruction_losses.append(avg_rec_loss)
        kl_losses.append(avg_kl_loss)
        
        print(f'Epoch {epoch+1}:')
        print(f'  Total Loss: {avg_loss:.4f}')
        print(f'  Reconstruction Loss: {avg_rec_loss:.4f}')
        print(f'  KL Loss: {avg_kl_loss:.4f}')
        
        if (epoch + 1) % 5 == 0:
            with torch.no_grad():
                reconstructed = model.decode(
                    model.reparameterize(q_mu[:8], q_logvar[:8]),
                    states[:8]
                )
                
                plt.figure(figsize=(12, 3))
                for i in range(8):
                    plt.subplot(2, 8, i+1)
                    plt.imshow(images[i].cpu().squeeze(), cmap='gray')
                    plt.axis('off')
                    plt.title(f'Label: {labels[i].item()}')
                    
                    plt.subplot(2, 8, i+9)
                    plt.imshow(reconstructed[i].cpu().squeeze(), cmap='gray')
                    plt.axis('off')
                    plt.title('Reconstructed')
                plt.tight_layout()
                plt.show()
    
    return model, train_losses, reconstruction_losses, kl_losses