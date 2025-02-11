import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import CircleDataset
from model import NaiveUnet, save_model


def train_model(model, dataloader, epochs=5, lr=1e-5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    total_batches = len(dataloader) * epochs

    with tqdm(total=total_batches, desc="Training") as pbar:
        for epoch in range(epochs):
            total_loss = 0
            for batch_idx, (noisy_images, noise, t) in enumerate(dataloader):
                noisy_images, noise, t = (
                    noisy_images.to(device),
                    noise.to(device),
                    t.to(device),
                )
                optimizer.zero_grad()
                predicted_noise = model(noisy_images, t)
                loss = loss_fn(predicted_noise, noise)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

                pbar.set_description_str(f"Training Epoch {epoch + 1}/{epochs}")
                pbar.update(1)

            avg_loss = total_loss / len(dataloader)  # Compute average loss per epoch
            print(f"Epoch {epoch + 1}/{epochs} - Loss: {avg_loss:.6f}")

    return model


def main():
    dataset = CircleDataset(num_samples=50000, image_size=32, T=100)
    dataloader = DataLoader(dataset, batch_size=256, shuffle=True)
    model = NaiveUnet(in_channels=1, out_channels=1, n_feat=256)
    try:
        model = train_model(model, dataloader, epochs=100)

    except Exception as e:
        print(e)
    finally:
        save_model(model, "model2.pth")


if __name__ == "__main__":
    main()
