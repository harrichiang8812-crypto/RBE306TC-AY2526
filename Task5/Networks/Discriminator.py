import sys
import torch
import torch.nn as nn

# Add the path for additional modules (ensure these paths are correct in your project)
sys.path.append('./')
from Networks.PlainGenerators.PlainGeneratorBlocks import Convblock

class Critic(nn.Module):
    def __init__(self, normalization='none', input_channels=1, generator_dim=64):
        """
        Args:
            normalization: The type of normalization to apply ('none' for WGAN, or instance/batchnorm for variants).
            input_channels: Number of input channels (e.g., 1 for grayscale).
            generator_dim: Base number of feature maps for the critic (default: 64).
        """
        super(Critic, self).__init__()

        # Convolutional layers for feature extraction
        self.conv1 = nn.Conv2d(input_channels, generator_dim, kernel_size=5, stride=2, padding=2)
        self.conv2 = Convblock(generator_dim, generator_dim * 2, normalization=normalization)
        self.conv3 = Convblock(generator_dim * 2, generator_dim * 4, normalization=normalization)
        self.conv4 = Convblock(generator_dim * 4, generator_dim * 8, normalization=normalization)
        self.conv5 = Convblock(generator_dim * 8, generator_dim * 8, normalization=normalization)

        # Final output: a single scalar score
        self.fc = nn.Linear(generator_dim * 8 * 2 * 2, 1)

    def forward(self, input):
        """
        Forward pass through the critic network.

        Args:
            input: Input tensor (e.g., an image or feature map).

        Returns:
            critic_score: A scalar value representing the "realness" of the input.
        """
        x = self.conv1(input)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)

        x = torch.flatten(x, 1)
        critic_score = self.fc(x)

        return critic_score