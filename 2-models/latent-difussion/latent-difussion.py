import torch
import torch.nn as nn
import torch.nn.functional as F

# Latent Difussion Model Encoder (Convert data into flatten with essential data only)
class LDME(nn.Module):
    def __init__(self, latent_dim=128):
        super(LDME, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1)
        self.fc = nn.Linear(128 * 8 * 8, latent_dim)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# Latent Difussion Model Decorder (Convert data back into orginal shape with noises)
class LDMD(nn.Module):
    def __init__(self, latent_dim=128):
        super(LDMD, self).__init__()
        self.fc = nn.Linear(latent_dim, 128 * 8 * 8)
        self.conv1 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.ConvTranspose2d(in_channels=64, out_channels=3, kernel_size=4, stride=2, padding=1)

    def forward(self, z):
        z = F.relu(self.fc(z))
        z = z.view(z.size(0), 128, 8, 8)
        z = F.relu(self.conv1(z))
        z = F.sigmoid(self.conv2(z))
        return z