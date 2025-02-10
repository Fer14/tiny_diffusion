import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class CircleDataset(Dataset):
    def __init__(self, num_samples=1000, image_size=32, T=100):
        self.num_samples = num_samples
        self.image_size = image_size
        self.beta_schedule = np.linspace(0.0001, 0.02, T)
        self.T = T

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        t = np.random.randint(0, self.T)
        x_0 = self.create_circle_image()
        x_t, noise = self.forward_diffusion(x_0, t)
        x_t = torch.tensor(x_t, dtype=torch.float32).unsqueeze(0)
        noise = torch.tensor(noise, dtype=torch.float32).unsqueeze(0)
        return x_t, noise, t

    def forward_diffusion(self, x_0, t):
        alpha_t = np.cumprod(1 - self.beta_schedule[: t + 1])[-1]
        noise = np.random.normal(0, 1, x_0.shape)
        x_t = np.sqrt(alpha_t) * x_0 + np.sqrt(1 - alpha_t) * noise
        return x_t, noise

    def create_circle_image(self):
        image = np.ones((self.image_size, self.image_size), dtype=np.uint8) * 255
        radius = np.random.randint(5, 15)
        center_x = np.random.randint(radius, self.image_size - radius)
        center_y = np.random.randint(radius, self.image_size - radius)
        image = cv2.circle(image, (center_x, center_y), radius, (0,), -1)
        return image / 255.0
