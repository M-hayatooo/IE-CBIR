import torch
import torch.nn as nn
import torch.nn.functional as F


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
        vec = self.reparameterize(mu, logvar)
        x_hat = self.decode(vec)
        return x_hat, vec, mu, logvar


class ResNetVAEencoder(nn.Module): # first_ch=16 second_ch=32 third_ch=64 forth_ch=128
    def __init__(self, first_ch, second_ch, third_ch, forth_ch, z_ch, ):
        super(ResNetVAEencoder, self).__init__()
        self.forth_ch = forth_ch
        self.block1 = nn.Sequential(
            nn.Conv3d(1, first_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm3d(first_ch),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(first_ch, first_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm3d(first_ch),
            nn.LeakyReLU(0.2, inplace=True),
        ) # here excurt pooling
        self.block2 = nn.Sequential(
            nn.Conv3d(first_ch, first_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm3d(first_ch),
            nn.LeakyReLU(0.2, inplace=True),
            # this phase channel up
            nn.Conv3d(first_ch, second_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm3d(second_ch),
            nn.LeakyReLU(0.2, inplace=True),
        ) # here excurt pooling
        self.block3 = nn.Sequential(
            nn.Conv3d(second_ch, second_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm3d(second_ch),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(second_ch, third_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm3d(third_ch),
            nn.LeakyReLU(0.2, inplace=True),
        ) ######## here excurt pooling ########
        self.block4short = nn.Sequential(
            nn.Conv3d(third_ch, third_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm3d(third_ch),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.block5 = nn.Sequential(
            nn.Conv3d(third_ch, third_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm3d(third_ch),
            nn.LeakyReLU(0.2, inplace=True),
            # nn.LeakyReLU(0.2, inplace=True), ここでスキップコネクション
        )
        self.block6 = nn.Sequential(
            nn.Conv3d(third_ch, third_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm3d(third_ch),
            nn.LeakyReLU(0.2, inplace=True),
            nn.AvgPool3d(kernel_size=2),
            nn.Conv3d(third_ch, forth_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm3d(forth_ch),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.block7 = nn.Sequential(
            nn.Conv3d(forth_ch, forth_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm3d(forth_ch),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(forth_ch, forth_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm3d(forth_ch),
            # nn.LeakyReLU(0.2, inplace=True),
        )
        self.block8 = nn.Sequential(
            nn.Conv3d(third_ch, third_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm3d(third_ch),
            nn.LeakyReLU(0.2, inplace=True),
            # this phase channel up
            nn.Conv3d(third_ch, forth_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm3d(forth_ch),
            nn.LeakyReLU(0.2, inplace=True),
        )
        
        self.pool1 = nn.AvgPool3d(kernel_size=2)
        self.pool2 = nn.AvgPool3d(kernel_size=2)
        self.pool3 = nn.AvgPool3d(kernel_size=2)
        self.pool4 = nn.AvgPool3d(kernel_size=2)
        self.Leakyrelu5 = nn.LeakyReLU(0.2, inplace=True)
        self.Leakyrelu7 = nn.LeakyReLU(0.2, inplace=True)
        # self.relu = nn.ReLU()
        ## 低次元空間のサイズ設定
        self.fc = nn.Linear(forth_ch*5*7*5, z_ch*2)
        n_classes = 3
        self.Wstar_layer = nn.Linear(z_ch, n_classes, bias=False)  # 'bias=False' allows the weights to be the pure representation vectors (not affected by the bias)


        
    def forward(self, x):
        x = self.block1(x)
        x = self.pool1(x) # Avgpooling  40
        x = self.block2(x) #32
        x = self.pool2(x) # Avgpooling  20
        x = self.block3(x)# 64
        x = self.pool3(x) # Avgpooling  10
        x = self.block4short(x) # 128
        h = self.block5(x) ###  skip  ###
        x = self.Leakyrelu5(x+h) # here add
        x = self.block6(x)  # Avgpooling  5
        h = self.block7(x) # conv→relu→conv
        x = self.Leakyrelu7(x+h) # here add
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        mu, logvar = x.chunk(2, dim=1)

        norm_z = mu / torch.norm(mu, p=2, dim=1, keepdim=True).detach() # 2)

        Wstar = self.Wstar_layer.weight.T#.detach()  # Note `.detach()`
        norm_Wstar = Wstar / torch.norm(Wstar, p=2, dim=0, keepdim=True)
        cosine_similarities = torch.mm(norm_z, norm_Wstar)

        return mu, logvar, cosine_similarities


class ResNetDecoder(nn.Module):
    def __init__(self, first_ch, second_ch, third_ch, forth_ch, z_ch):
        super(ResNetDecoder, self).__init__()
        self.forth_ch = forth_ch
        self.dfc = nn.Sequential(
            ##  低次元空間のサイズを決める
            nn.Linear(z_ch, forth_ch*175), # 5*6*5 = 150
            nn.ReLU(True),
        )
        self.block1 = nn.Sequential(
            nn.Conv3d(forth_ch, forth_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm3d(forth_ch),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(forth_ch, forth_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm3d(forth_ch),
            # nn.LeakyReLU(0.2, inplace=True),
        )
        self.block2u = nn.Sequential(
            nn.Conv3d(forth_ch, forth_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm3d(forth_ch),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2, mode='nearest'),
            # nn.Upsample(scale_factor=2, mode="trilinear", align_corners=True),
            nn.Conv3d(forth_ch, third_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm3d(third_ch),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.block3 = nn.Sequential(
            nn.Conv3d(third_ch, third_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm3d(third_ch),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(third_ch, third_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm3d(third_ch),
            # nn.LeakyReLU(0.2, inplace=True),
        )
        self.block4u = nn.Sequential(
            nn.Conv3d(third_ch, third_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm3d(third_ch),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2, mode='nearest'),
            # nn.Upsample(scale_factor=2, mode="trilinear", align_corners=True),
            nn.Conv3d(third_ch, second_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm3d(second_ch),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.block5u = nn.Sequential(
            nn.Conv3d(second_ch, second_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm3d(second_ch),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2, mode='nearest'),
            # nn.Upsample(scale_factor=2, mode="trilinear", align_corners=True),
            nn.Conv3d(second_ch, first_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm3d(first_ch),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.block6u = nn.Sequential(
            nn.Conv3d(first_ch, first_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm3d(first_ch),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2, mode='nearest'),
            # nn.Upsample(scale_factor=2, mode="trilinear", align_corners=True),
            nn.Conv3d(first_ch, first_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm3d(first_ch),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.last_block = nn.Sequential(
            nn.Conv3d(first_ch, 1, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(),
            # nn.Softmax(),
        )
        self.dLeakyrelu1 = nn.LeakyReLU(0.2, inplace=True)
        self.dLeakyrelu2 = nn.LeakyReLU(0.2, inplace=True)
        
    def forward(self, z):
        y = z.view(z.size(0), -1)
        y = self.dfc(y)
        y = y.view(y.size(0), self.forth_ch, 5, 7, 5)
        h = self.block1(y) # ------ skip
        y = self.dLeakyrelu1(y+h)
        y = self.block2u(y) ######--10*12*10
        h = self.block3(y) # ------ skip
        y = self.dLeakyrelu2(y+h)
        y = self.block4u(y) ######--20*24*20
        y = self.block5u(y) ######--40*48*40
        y = self.block6u(y) ######--80*96*80
        y = self.last_block(y)
        return y


class ResNetVAE(BaseVAE):
    def __init__(self, first_ch, second_ch, third_ch, forth_ch, z_ch) -> None:
        super(ResNetVAE, self).__init__()
        self.encoder = ResNetVAEencoder(first_ch, second_ch, third_ch, forth_ch, z_ch)
        self.decoder = ResNetDecoder(first_ch, second_ch, third_ch, forth_ch, z_ch)

    def reparamenterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar, cos_sim = self.encoder(x)
        z = self.reparamenterize(mu, logvar)
        x_re = self.decoder(z)
        return x_re, mu, logvar, cos_sim
