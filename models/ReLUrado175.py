from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


class BuildingBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride, bias=False):
        super(BuildingBlock, self).__init__()
        self.res = stride == 1
#        self.shortcut = self._shortcut()
        self.shortcut = self._shortcut(in_ch, out_ch)
        self.relu = nn.ReLU(inplace=True) # nn.ReLU(inplace=True)
        self.block = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=bias),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True), # nn.ReLU(inplace=True),
            nn.AvgPool3d(kernel_size=stride),
            nn.Conv3d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=bias),
            nn.BatchNorm3d(out_ch),
        )

    # def _shortcut(self):
    #     return lambda x: x

    def _shortcut(self, in_ch, out_ch):
        if in_ch != out_ch:
            return self._projection(in_ch, out_ch)
        else:
            return lambda x: x

    def _projection(self, channel_in, channel_out):
        return nn.Conv3d(channel_in, channel_out, kernel_size=1, stride=1, padding=0)


    def forward(self, x):
        if self.res:
            shortcut = self.shortcut(x)
            return self.relu(self.block(x) + shortcut)
        else:
            return self.relu(self.block(x))


class UpsampleBuildingkBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride, bias=False):
        super(UpsampleBuildingkBlock, self).__init__()
        self.res = stride == 1
        # self.shortcut = self._shortcut()
        self.shortcut = self._shortcut(in_ch, out_ch)
        self.relu = nn.ReLU(inplace=True) # nn.ReLU(inplace=True)
        self.block = nn.Sequential(
            nn.Conv3d(in_ch, in_ch, kernel_size=3, stride=1, padding=1, bias=bias),
            nn.BatchNorm3d(in_ch),
            nn.ReLU(inplace=True), # nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=stride),
            nn.Conv3d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=bias),
            nn.BatchNorm3d(out_ch),
        )

    # def _shortcut(self):
    #     return lambda x: x

    def _shortcut(self, in_ch, out_ch):
        if in_ch != out_ch:
            return self._projection(in_ch, out_ch)
        else:
            return lambda x: x

    def _projection(self, channel_in, channel_out):
        return nn.Conv3d(channel_in, channel_out, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        if self.res:
            shortcut = self.shortcut(x)
            return self.relu(self.block(x) + shortcut)
        else:
            return self.relu(self.block(x))


class ResNetEncoder(nn.Module):
    def __init__(self, in_ch, block_setting):
        super(ResNetEncoder, self).__init__()
        self.block_setting = block_setting
        self.in_ch = in_ch
        last = 1
        blocks = [nn.Sequential(
            nn.Conv3d(1, in_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm3d(in_ch),
            nn.ReLU(inplace=True),
        )]
        for line in self.block_setting:
            c, n, s = line[0], line[1], line[2]
            for i in range(n):
                stride = s if i == 0 else 1
                blocks.append(nn.Sequential(BuildingBlock(in_ch, c, stride)))
                in_ch = c
        self.inner_ch = in_ch
        self.blocks = nn.Sequential(*blocks)
        self.conv = nn.Sequential(nn.Conv3d(in_ch, last, kernel_size=1, stride=1, bias=True))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.blocks(x)
        return self.conv(h)

class ResNetDecoder(nn.Module):
    def __init__(self, encoder: ResNetEncoder, blocks=None):
        super(ResNetDecoder, self).__init__()
        last = encoder.block_setting[-1][0]
        if blocks is None:
            blocks = [nn.Sequential(
                nn.Conv3d(1, last, 1, 1, bias=True),
                nn.BatchNorm3d(last),
                nn.ReLU(inplace=True),
            )]
        in_ch = last
        for i in range(len(encoder.block_setting)):
            if i == len(encoder.block_setting) - 1:
                nc = encoder.in_ch
            else:
                nc = encoder.block_setting[::-1][i + 1][0]
            c, n, s = encoder.block_setting[::-1][i]
            for j in range(n):
                stride = s if j == n - 1 else 1
                c = nc if j == n - 1 else c
                blocks.append(nn.Sequential(UpsampleBuildingkBlock(in_ch, c, stride)))
                in_ch = c
        blocks.append(nn.Sequential(
            nn.Conv3d(in_ch, 1, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(),
            # nn.LeakyReLU(0.2),
            # nn.Softmax(dim=1),
        ))
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(x)


class BaseEncoder(nn.Module):
    def __init__(self) -> None:
        super(BaseEncoder, self).__init__()
class BaseDecoder(nn.Module):
    def __init__(self) -> None:
        super(BaseDecoder, self).__init__()

class BaseCAE(nn.Module):
    def __init__(self) -> None:
        super(BaseCAE, self).__init__()
        self.encoder = BaseEncoder()
        self.decoder = BaseDecoder()
    def encode(self, x):
        z = self.encoder(x)
        return z
    def decode(self, z):
        out = self.decoder(z)
        return out
    def forward(self, x):
        z = self.encode(x)
        out = self.decode(z)
        return out, z


class ResNetCAE(BaseCAE):
    def __init__(self, in_ch, block_setting) -> None:
        super(ResNetCAE, self).__init__()
        self.encoder = ResNetEncoder(
            in_ch=in_ch,
            block_setting=block_setting,
        )
        self.decoder = ResNetDecoder(self.encoder)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def __call__(self, x):
        x = self.forward(x)
        return x


class BaseVAE(nn.Module):
    def __init__(self) -> None:
        super(BaseVAE, self).__init__()
        self.encoder = BaseEncoder()
        self.decoder = BaseDecoder()
    def encode(self, x):
        mu, logvar = self.encoder(x)
        return mu, logvar
    def decode(self, vec):
        out = self.decoder(vec)
        return out
    def reparameterize(self, mu, logvar) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decode(z)
        return x_hat, z, mu, logvar


class VAEResNetEncoder(ResNetEncoder):
    def __init__(self, in_ch, block_setting) -> None:
        super(VAEResNetEncoder, self).__init__(in_ch, block_setting)
        self.mu = nn.Conv3d(self.inner_ch, 1, kernel_size=1, stride=1, bias=True)
        self.var = nn.Conv3d(self.inner_ch, 1, kernel_size=1, stride=1, bias=True)
        # self.fc = nn.Linear(self.inner_ch, 3)
        # self.fc = nn.Linear(forth_ch*5*7*5, z_ch*2)
        # self.down = nn.Linear(z_ch, 50)
        self.Wstar_layer = nn.Linear(175, 2, bias=False)  # 'bias=False' allows the weights to be the pure representation vectors (not affected by the bias)

    def forward(self, x: torch.Tensor):
        h = self.blocks(x) # h = batch_size*64*5*7*5
        mu = self.mu(h) # mu = batch_size*1*5*7*5
        logvar = self.var(h)
        b_size = mu.size(0)
        z = mu.view(b_size, -1)
        norm_z = z / torch.norm(z, p=2, dim=1, keepdim=True)
        Wstar = self.Wstar_layer.weight.T # .detach() # Note '.detach()'  ※ Wstarをdetachしないと精度が上がる
        # detachしてしまうと、fine-tuning時にWstarのバックプロップできない
        norm_Wstar = Wstar / torch.norm(Wstar, p=2, dim=0, keepdim=True)
        cosin_similarities = torch.mm(norm_z, norm_Wstar)
        return mu, logvar, cosin_similarities


class RaDOBaseVAE(BaseVAE):
    def __init__(self, in_ch, block_setting) -> None:
        super(RaDOBaseVAE, self).__init__()
        self.encoder = VAEResNetEncoder(in_ch=in_ch, block_setting=block_setting)
        self.decoder = ResNetDecoder(self.encoder)

    def reparamenterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar, cos_sim = self.encoder(x)
        # proto = mu.view(mu.size(0), -1)
        z = self.reparamenterize(mu, logvar)
        x_rec_z = self.decoder(z)
        x_rec_mu = self.decoder(mu)
        return x_rec_mu, x_rec_z, mu, logvar, cos_sim
