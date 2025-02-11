import torch
import torch.nn as nn


class Conv3(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, is_res: bool = False
    ) -> None:
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, groups=in_channels),
            nn.Conv2d(out_channels, out_channels, 1),
            nn.GroupNorm(4, out_channels),
            nn.ReLU(),
        )
        self.conv = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, groups=out_channels),
            nn.Conv2d(out_channels, out_channels, 1),
            nn.GroupNorm(4, out_channels),
            nn.ReLU(),
        )

        self.is_res = is_res

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.main(x)
        if self.is_res:
            x = x + self.conv(x)
            return x / 1.414
        else:
            return self.conv(x)


class UnetDown(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super(UnetDown, self).__init__()
        layers = (Conv3(in_channels, out_channels), nn.MaxPool2d(2))
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class UnetUp(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super(UnetUp, self).__init__()
        layers = (
            nn.ConvTranspose2d(in_channels, out_channels, 2, 2),
            Conv3(out_channels, out_channels),
        )
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = torch.cat((x, skip), 1)
        x = self.model(x)
        return x


class TimeSiren(nn.Module):
    def __init__(self, emb_dim: int) -> None:
        super(TimeSiren, self).__init__()

        self.lin1 = nn.Linear(1, emb_dim, bias=False)
        self.lin2 = nn.Linear(emb_dim, emb_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(-1, 1)
        x = torch.sin(self.lin1(x))
        x = self.lin2(x)
        return x


class NaiveUnet(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, n_feat: int = 256) -> None:
        super(NaiveUnet, self).__init__()
        self.n_feat = n_feat

        self.init_conv = Conv3(in_channels, n_feat, is_res=True)

        self.down1 = UnetDown(n_feat, n_feat)
        self.down2 = UnetDown(n_feat, 2 * n_feat)
        self.down3 = UnetDown(2 * n_feat, 2 * n_feat)

        self.to_vec = nn.AdaptiveAvgPool2d(1)

        self.timeembed = TimeSiren(2 * n_feat)

        self.up0 = nn.Sequential(
            nn.ConvTranspose2d(2 * n_feat, 2 * n_feat, 4, 4),
            nn.GroupNorm(4, 2 * n_feat),
            nn.ReLU(),
        )

        self.up1 = UnetUp(4 * n_feat, 2 * n_feat)
        self.up2 = UnetUp(4 * n_feat, n_feat)
        self.up3 = UnetUp(2 * n_feat, n_feat)
        self.out = nn.Conv2d(2 * n_feat, out_channels, 3, 1, 1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # X.shape()  = B, 1,32,32
        x = self.init_conv(x)  # B, 256, 32, 32

        down1 = self.down1(x)  # B, 256, 16,16
        down2 = self.down2(down1)  # B, 512, 8,8
        down3 = self.down3(down2)  # B, 512, 4,4

        thro = self.to_vec(down3)  # B, 5125, 1,1

        t = t.float()  # t.shape() = B
        temb = self.timeembed(t).view(-1, self.n_feat * 2, 1, 1)  # B, 5125, 1,1

        thro_emb = thro + temb  # B, 5125, 1,1

        thro = self.up0(thro_emb)  # B, 5125, 4,4

        up1 = self.up1(thro, down3) + temb  # B, 5125, 8,8
        up2 = self.up2(up1, down2)  # B, 256, 16,16
        up3 = self.up3(up2, down1)  # B, 256, 32,32

        out = self.out(torch.cat((up3, x), 1))  # B, 1, 32,32

        return out


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
