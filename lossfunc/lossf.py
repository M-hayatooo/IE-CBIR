import torch
import torch.nn as nn
import torch.nn.functional as F


#  mean squared error
def mse_loss(out, x):
    bsize = x.size(0)
    x = x.view(bsize, -1)
    out = out.view(bsize, -1)
    loss = torch.mean(torch.sum(F.mse_loss(x, out, reduction='none'), dim=1), dim=0)
    voxel_size = x.shape[1]
    mse = loss / voxel_size
    return mse

# squared error
def squared_error(out, x):
    bsize = x.size(0)
    x = x.view(bsize, -1)
    out = out.view(bsize, -1)
    se = torch.mean(torch.sum(F.mse_loss(x, out, reduction='none'), dim=1), dim=0)
    return se


def kld_loss(mu, logvar):
    bsize = mu.size(0)
    mu = mu.view(bsize, -1)
    logvar = logvar.view(bsize, -1)
    return torch.mean(-0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp(), dim=1), dim=0)
 
#                                      1 + var-tf.square(mean)(==^2)-tf.exp(var)
#                     -0.5 * tf.reduce_sum(kl_div_loss, 1)
#        tf.reduce_mean(kl_div_loss)

def RaDOBase_loss(x_rec_mu, x_rec_z, mu, logvar, cos_loss, inputs, d1_w, d2_w, kl_w):
    mse = mse_loss(x_rec_mu, inputs)
    se = squared_error(x_rec_mu, inputs)
    d1 = se
    d2 = squared_error(x_rec_mu, x_rec_z)
    kld = kld_loss(mu, logvar)
    loss = d1_w*d1 + d2_w*d2 + kl_w*kld + cos_loss
    return loss, se, mse, kld, d1, d2


def VAEBase_loss(x_rec_z, mu, logvar, cos_loss, inputs, d1_w, kl_w):
    mse = mse_loss(x_rec_z, inputs)
    se = squared_error(x_rec_z, inputs)
    d1 = se
    kld = kld_loss(mu, logvar)
    loss = d1_w*d1 + kl_w*kld + cos_loss
    return loss, se, mse, kld, d1


def CAEBase_loss(x_rec_z, cos_loss, inputs, d1_w):
    mse = mse_loss(x_rec_z, inputs)
    se = squared_error(x_rec_z, inputs)
    d1 = se
    loss = d1_w*d1 + cos_loss
    return loss, se, mse, d1


def localized_loss(x_hat, mu, logvar, localize_loss, x, msew=1, kldw=1, localizew=1):
    mse = mse_loss(x_hat, x) * msew
    kld = kld_loss(mu, logvar) * kldw
    localize_loss = torch.mean(torch.sum(localize_loss, dim=1), dim=0) * localizew
    loss = mse + kld + localize_loss
    return loss, mse, kld, localize_loss
