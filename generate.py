import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import ListedColormap

from dataset import CircleDataset
from model import NaiveUnet, load_model

colors = [
    (255 / 255, 255 / 255, 255 / 255),
    (247 / 255, 178 / 255, 103 / 255),
    (255 / 255, 243 / 255, 176 / 255),
    (240 / 255, 45 / 255, 58 / 255),
]


custom_cmap = ListedColormap(colors, name="custom")


def sample(model, dataset, num_samples=1, T=1000):
    # Initialize random noise (final timestep)
    device = next(model.parameters()).device  # Use the device of the model
    x_t = torch.randn(num_samples, 1, dataset.image_size, dataset.image_size).to(device)

    # Reverse the diffusion process
    for t in reversed(range(T)):
        # Get the predicted noise from the model
        with torch.no_grad():
            predicted_noise = model(x_t, torch.tensor([t], device=device))

        # Get the beta schedule and alpha_t at timestep t
        beta_t = dataset.beta_schedule[t]
        alpha_t = np.prod(1 - dataset.beta_schedule[: t + 1])

        alpha_t = torch.tensor(alpha_t, dtype=torch.float32).to(device)
        beta_t = torch.tensor(beta_t, dtype=torch.float32).to(device)

        # Update x_t to the denoised image at timestep t-1
        if t > 0:
            noise = torch.randn_like(x_t)
            x_t = (1 / torch.sqrt(1 - beta_t)) * (
                x_t - beta_t / torch.sqrt(1 - alpha_t) * predicted_noise
            ) + torch.sqrt(beta_t) * noise
        else:
            # At the final timestep (t=0), no noise is added.
            x_t = (1 / torch.sqrt(1 - beta_t)) * (
                x_t - beta_t / torch.sqrt(1 - alpha_t) * predicted_noise
            )

    # The final x_t should be the denoised image
    return x_t


model = NaiveUnet(in_channels=1, out_channels=1, n_feat=256)


model = load_model(model, "model2.pth", device="cuda")
t = 100

dataset = CircleDataset(num_samples=1, image_size=32, T=t)

model.eval()

# Sample a new image from the trained model
num_samples = 1  # Number of images to generate
samples = sample(model, dataset, num_samples=num_samples, T=t)

# Plot the generated image
generated_image = samples.squeeze().cpu().numpy()

plt.imshow(generated_image, cmap=custom_cmap)

plt.axis("off")  # Remove axes for a clean image

# Save the plot as an image
plt.savefig("generated_im.png", bbox_inches="tight", pad_inches=0, dpi=300)
plt.close()  # Close the figure to free memory
