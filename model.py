import torch
import torch.nn as nn


class EmbedBlock(nn.Module):
    def __init__(self, input_dim, emb_dim):
        super().__init__()
        self.input_dim = input_dim
        layers = [
            nn.Linear(input_dim, emb_dim),
            nn.ReLU(),
            nn.Linear(emb_dim, emb_dim),
            nn.Unflatten(1, (emb_dim, 1, 1)),  # Reshape to (batch_size, emb_dim, 1, 1)
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, input):
        input = input.view(-1, self.input_dim)  # Ensure input is flattened
        return self.model(input)


class UNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNetBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        return x


class TinyUNet(nn.Module):
    def __init__(self, in_channels=1, base_channels=32, timestep_embed_dim=512):
        super(TinyUNet, self).__init__()
        self.timestep_embed_dim = timestep_embed_dim

        # Encoder
        self.enc1 = UNetBlock(in_channels, base_channels)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = UNetBlock(base_channels, base_channels * 2)
        self.pool2 = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = UNetBlock(base_channels * 2, base_channels * 4)

        # Decoder
        self.up2 = nn.ConvTranspose2d(
            base_channels * 4, base_channels * 2, kernel_size=2, stride=2
        )
        self.dec2 = UNetBlock(base_channels * 4, base_channels * 2)
        self.up1 = nn.ConvTranspose2d(
            base_channels * 2, base_channels, kernel_size=2, stride=2
        )
        self.dec1 = UNetBlock(base_channels * 2, base_channels)

        # Final Output Layer
        self.final_conv = nn.Conv2d(base_channels, in_channels, kernel_size=1)

        # Time embedding block
        self.t_embed = EmbedBlock(input_dim=1, emb_dim=timestep_embed_dim)
        # Projection layers for timestep embedding
        self.t_embed_proj_bottleneck = nn.Linear(timestep_embed_dim, base_channels * 4)
        self.t_embed_proj_decoder = nn.Linear(timestep_embed_dim, base_channels * 2)

    def forward(self, x, t):
        # Convert t to float32
        t = t.float()  # Ensure t is of type torch.float32

        # Get timestep embedding
        t_embed = self.t_embed(t)  # Shape: (batch_size, timestep_embed_dim, 1, 1)

        # Encoder
        x1 = self.enc1(x)
        x2 = self.pool1(x1)
        x3 = self.enc2(x2)
        x4 = self.pool2(x3)

        # Bottleneck
        x5 = self.bottleneck(x4)
        # Project and add timestep embedding
        t_embed_bottleneck = self.t_embed_proj_bottleneck(
            t_embed.squeeze(-1).squeeze(-1)
        )  # Remove spatial dims
        t_embed_bottleneck = t_embed_bottleneck.view(
            -1, self.t_embed_proj_bottleneck.out_features, 1, 1
        )
        x5 = x5 + t_embed_bottleneck

        # Decoder
        x6 = self.up2(x5)
        x6 = torch.cat([x6, x3], dim=1)  # Skip Connection
        x7 = self.dec2(x6)
        # Project and add timestep embedding
        t_embed_decoder = self.t_embed_proj_decoder(
            t_embed.squeeze(-1).squeeze(-1)
        )  # Remove spatial dims
        t_embed_decoder = t_embed_decoder.view(
            -1, self.t_embed_proj_decoder.out_features, 1, 1
        )
        x7 = x7 + t_embed_decoder

        x8 = self.up1(x7)
        x8 = torch.cat([x8, x1], dim=1)  # Skip Connection
        x9 = self.dec1(x8)

        # Final Output
        output = self.final_conv(x9)
        return output


def save_model(model, file_path):
    # Move the model to CPU before saving, if it is on GPU
    model.to("cpu")
    torch.save(model.state_dict(), file_path)
    print(f"Model saved to {file_path}")


def load_model(model, file_path, device):
    state_dict = torch.load(file_path)

    # Load the state_dict into the model
    model.load_state_dict(state_dict)

    # Move model to the specified device
    model.to(device)

    # Set the model to evaluation mode
    model.eval()

    return model
