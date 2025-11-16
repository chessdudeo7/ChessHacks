import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    """
    A simple residual block with two convolutional layers.
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, padding='same'):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        identity = x # Save the input
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity # Add the input back (the "residual" connection)
        return F.relu(out)

class ChessNet(nn.Module):
    """
    A CNN-based model for chess move prediction (Policy-only).
    """
    def __init__(self, num_residual_blocks=4, num_filters=64, output_size=10000):
        super().__init__()
        
        # --- CNN Body ---
        # Initial convolutional layer to create the filter depth
        self.entry_conv = nn.Conv2d(
            in_channels=12,  # 12 piece planes (P,N,B,R,Q,K for white/black)
            out_channels=num_filters, 
            kernel_size=3, 
            padding='same'
        )
        self.entry_bn = nn.BatchNorm2d(num_filters)

        # A series of residual blocks
        self.residual_blocks = nn.ModuleList(
            [ResidualBlock(num_filters, num_filters) for _ in range(num_residual_blocks)]
        )

        # --- Policy Head ---
        # The CNN body's output (num_filters, 8, 8) is flattened
        # and combined with the 5 scalar features.
        cnn_output_size = num_filters * 8 * 8  # 64 * 64 = 4096
        scalar_input_size = 5 # Castling rights, side to move
        combined_input_size = cnn_output_size + scalar_input_size

        self.policy_fc1 = nn.Linear(combined_input_size, 512)
        self.policy_fc2 = nn.Linear(512, 512)
        self.policy_fc3 = nn.Linear(512, output_size)

    def forward(self, x_spatial, x_scalar):
        """
        x_spatial: (batch_size, 12, 8, 8) - The board state
        x_scalar: (batch_size, 5) - Castling rights, side to move
        """
        
        # Process spatial data through CNNs
        out = F.relu(self.entry_bn(self.entry_conv(x_spatial)))
        for block in self.residual_blocks:
            out = block(out)

        # Flatten the CNN output
        out_flat = out.view(out.size(0), -1) # (batch_size, num_filters * 8 * 8)

        # Combine with scalar features
        combined = torch.cat([out_flat, x_scalar], dim=1)

        # Pass through Policy Head
        p = F.relu(self.policy_fc1(combined))
        p = F.relu(self.policy_fc2(p))
        p_logits = self.policy_fc3(p) # Raw logits for CrossEntropyLoss

        return p_logits