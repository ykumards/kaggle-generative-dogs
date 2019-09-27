import torch
from torch import nn, optim
import torch.nn.functional as F


def init_weights(m):
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.)

# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def snconv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
    return nn.utils.spectral_norm(nn.Conv2d(in_channels=in_channels, 
                                            out_channels=out_channels, 
                                            kernel_size=kernel_size,
                                            stride=stride, 
                                            padding=padding, 
                                            dilation=dilation, 
                                            groups=groups, 
                                            bias=bias))


def snlinear(in_features, out_features):
    return nn.utils.spectral_norm(nn.Linear(in_features=in_features, out_features=out_features))


def sn_embedding(num_embeddings, embedding_dim):
    return nn.utils.spectral_norm(nn.Embedding(num_embeddings=num_embeddings, embedding_dim=embedding_dim))

# ----------------------------------------------------------------------------
# Pixelwise feature vector normalization.
# reference: https://github.com/tkarras/progressive_growing_of_gans/blob/master/networks.py#L120
# ----------------------------------------------------------------------------
class PixelwiseNorm(nn.Module):
    def __init__(self):
        super(PixelwiseNorm, self).__init__()

    def forward(self, x, alpha=1e-8):
        """
        forward pass of the module
        :param x: input activations volume
        :param alpha: small number for numerical stability
        :return: y => pixel normalized activations
        """
        y = x.pow(2.).mean(dim=1, keepdim=True).add(alpha).sqrt()  # [N1HW]
        y = x / y  # normalize the input x volume
        return y
    
class MinibatchStdDev(nn.Module):
    """
    Minibatch standard deviation layer for the discriminator
    """

    def __init__(self):
        """
        derived class constructor
        """
        super(MinibatchStdDev, self).__init__()

    def forward(self, x, alpha=1e-8):
        """
        forward pass of the layer
        :param x: input activation volume
        :param alpha: small number for numerical stability
        :return: y => x appended with standard deviation constant map
        """
        batch_size, _, height, width = x.shape
        # [B x C x H x W] Subtract mean over batch.
        y = x - x.mean(dim=0, keepdim=True)
        # [1 x C x H x W]  Calc standard deviation over batch
        y = torch.sqrt(y.pow(2.).mean(dim=0, keepdim=False) + alpha)

        # [1]  Take average over feature_maps and pixels.
        y = y.mean().view(1, 1, 1, 1)

        # [B x 1 x H x W]  Replicate over group and pixels.
        y = y.repeat(batch_size,1, height, width)

        # [B x C x H x W]  Append as new feature_map.
        y = torch.cat([x, y], 1)
        # return the computed values:
        return y    
    
    
class GaussianNoise(nn.Module):
    """Gaussian noise regularizer.

    Args:
        sigma (float, optional): relative standard deviation used to generate the
            noise. Relative means that it will be multiplied by the magnitude of
            the value your are adding the noise to. This means that sigma can be
            the same regardless of the scale of the vector.
        is_relative_detach (bool, optional): whether to detach the variable before
            computing the scale of the noise. If `False` then the scale of the noise
            won't be seen as a constant but something to optimize: this will bias the
            network to generate vectors with smaller values.
    """

    def __init__(self, sigma=0.1, is_relative_detach=True):
        super().__init__()
        self.sigma = sigma
        self.is_relative_detach = is_relative_detach
        self.noise = torch.tensor(0.).to(device)

    def forward(self, x):
        if self.training and self.sigma != 0:
            scale = self.sigma * x.detach() if self.is_relative_detach else self.sigma * x
            sampled_noise = self.noise.repeat(*x.size()).normal_() * scale
            x = x + sampled_noise
        return x 

class ConditionalBatchNorm2d(nn.Module):
    def __init__(self, num_features, num_classes):
        super().__init__()
        self.num_features = num_features
        self.bn = nn.BatchNorm2d(num_features)
        self.embed = nn.Embedding(num_classes, num_features * 2)
        self.embed.weight.data[:, :num_features].fill_(1.)  # Initialize scale to 1
        self.embed.weight.data[:, num_features:].zero_()    # Initialize bias at 0

    def forward(self, x, y):
        out = self.bn(x)
        gamma, beta = self.embed(y).chunk(2, 1)
        out = gamma.view(-1, self.num_features, 1, 1) * out + beta.view(-1, self.num_features, 1, 1)
        return out    

class Self_Attn(nn.Module):
    """ Self attention Layer"""

    def __init__(self, in_channels):
        super(Self_Attn, self).__init__()
        self.in_channels = in_channels
        self.snconv1x1_theta = snconv2d(in_channels=in_channels, out_channels=in_channels//8, kernel_size=1, stride=1, padding=0)
        self.snconv1x1_phi = snconv2d(in_channels=in_channels, out_channels=in_channels//8, kernel_size=1, stride=1, padding=0)
        self.snconv1x1_g = snconv2d(in_channels=in_channels, out_channels=in_channels//2, kernel_size=1, stride=1, padding=0)
        self.snconv1x1_attn = snconv2d(in_channels=in_channels//2, out_channels=in_channels, kernel_size=1, stride=1, padding=0)
        self.maxpool = nn.MaxPool2d(2, stride=2, padding=0)
        self.softmax  = nn.Softmax(dim=-1)
        self.sigma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        """
            inputs :
                x : input feature maps(B X C X W X H)
            returns :
                out : self attention value + input feature 
                attention: B X N X N (N is Width*Height)
        """
        _, ch, h, w = x.size()
        # Theta path
        theta = self.snconv1x1_theta(x)
        theta = theta.view(-1, ch//8, h*w)
        # Phi path
        phi = self.snconv1x1_phi(x)
        phi = self.maxpool(phi)
        phi = phi.view(-1, ch//8, h*w//4)
        # Attn map
        attn = torch.bmm(theta.permute(0, 2, 1), phi)
        attn = self.softmax(attn)
        # g path
        g = self.snconv1x1_g(x)
        g = self.maxpool(g)
        g = g.view(-1, ch//2, h*w//4)
        # Attn_g
        attn_g = torch.bmm(g, attn.permute(0, 2, 1))
        attn_g = attn_g.view(-1, ch//2, h, w)
        attn_g = self.snconv1x1_attn(attn_g)
        # Out
        out = x + self.sigma*attn_g
        return out    

class GenBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_classes):
        super(GenBlock, self).__init__()
        self.cond_bn1 = ConditionalBatchNorm2d(in_channels, num_classes)
        self.relu = nn.ReLU(inplace=True)
        self.ps1 = nn.PixelShuffle(2),
        self.snconv2d1 = snconv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        self.cond_bn2 = ConditionalBatchNorm2d(out_channels, num_classes)
        self.snconv2d2 = snconv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        self.snconv2d0 = snconv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x, labels):
        x0 = x

        x = self.cond_bn1(x, labels)
        x = self.relu(x)
        x = F.interpolate(x, scale_factor=2, mode='nearest') # upsample
        x = self.snconv2d1(x)
        x = self.cond_bn2(x, labels)
        x = self.relu(x)
        x = self.snconv2d2(x)

        x0 = F.interpolate(x0, scale_factor=2, mode='nearest') # upsample
        x0 = self.snconv2d0(x0)

        out = x + x0
        return out
    
class DiscOptBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DiscOptBlock, self).__init__()
        self.snconv2d1 = snconv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.snconv2d2 = snconv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        self.downsample = nn.AvgPool2d(2)
        self.snconv2d0 = snconv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x0 = x

        x = self.snconv2d1(x)
        x = self.relu(x)
        x = self.snconv2d2(x)
        x = self.downsample(x)

        x0 = self.downsample(x0)
        x0 = self.snconv2d0(x0)

        out = x + x0
        return out


class DiscBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DiscBlock, self).__init__()
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.snconv2d1 = snconv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        self.snconv2d2 = snconv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        self.downsample = nn.AvgPool2d(2)
        self.ch_mismatch = False
        if in_channels != out_channels:
            self.ch_mismatch = True
        self.snconv2d0 = snconv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x, downsample=True):
        x0 = x

        x = self.relu(x)
        x = self.snconv2d1(x)
        x = self.relu(x)
#         x = self.snconv2d2(x)
        if downsample:
            x = self.downsample(x)

        if downsample or self.ch_mismatch:
            x0 = self.snconv2d0(x0)
            if downsample:
                x0 = self.downsample(x0)

        out = x + x0
        return out
    
class Generator(nn.Module):
    """Generator."""

    def __init__(self, args):
        super(Generator, self).__init__()

        self.z_dim = args.latent_dim
        self.g_conv_dim = args.num_feature_maps_gen
        
        self.snlinear0 = snlinear(in_features=self.z_dim, out_features=self.g_conv_dim*8*4*4)
        self.block1 = GenBlock(self.g_conv_dim*8, self.g_conv_dim*8, args.num_classes)
        self.block2 = GenBlock(self.g_conv_dim*8, self.g_conv_dim*4, args.num_classes)
        self.block3 = GenBlock(self.g_conv_dim*4, self.g_conv_dim*2, args.num_classes)
        self.self_attn = Self_Attn(self.g_conv_dim*2)
        self.block4 = GenBlock(self.g_conv_dim*2, self.g_conv_dim, args.num_classes)
#         self.block5 = GenBlock(self.g_conv_dim, self.g_conv_dim, args.num_classes)
        self.bn = nn.BatchNorm2d(self.g_conv_dim, eps=1e-5, momentum=0.0001, affine=True)
        self.relu = nn.ReLU(inplace=True)
        self.snconv2d1 = snconv2d(in_channels=self.g_conv_dim, out_channels=3, kernel_size=3, stride=1, padding=1)
        self.tanh = nn.Tanh()

        # Weight init
        self.apply(init_weights)

    def forward(self, z, labels):
#         import pdb; pdb.set_trace()
        # n x z_dim
        act0 = self.snlinear0(z)            # n x g_conv_dim*8*2*2
        act0 = act0.view(-1, self.g_conv_dim*8, 4, 4) # n x g_conv_dim*16 x 2 x 2
        act1 = self.block1(act0, labels)    # n x g_conv_dim*8 x 4 x 4
        act2 = self.block2(act1, labels)    # n x g_conv_dim*4 x 8 x 8
        act3 = self.block3(act2, labels)    # n x g_conv_dim*2 x 16 x 16
        act3 = self.self_attn(act3)         # n x g_conv_dim*2 x 16 x 16
        act4 = self.block4(act3, labels)    # n x g_conv_dim*1 x 32 x 32
#         act5 = self.block5(act4, labels)    # n x g_conv_dim  x 64 x 64
#         act5 = self.bn(act4)                # n x g_conv_dim  x 64 x 64
#         act5 = self.relu(act5)              # n x g_conv_dim  x 64 x 64
        act6 = self.snconv2d1(act4)         # n x 3 x 64 x 64
        act6 = self.tanh(act6)              # n x 3 x 64 x 64
        return act6

    
class Discriminator(nn.Module):
    """Discriminator."""

    def __init__(self, args):
        super(Discriminator, self).__init__()
        self.d_conv_dim = args.num_feature_maps_disc
        self.opt_block1 = DiscOptBlock(3, self.d_conv_dim)
        self.block1 = DiscBlock(self.d_conv_dim, self.d_conv_dim*2)
        self.self_attn = Self_Attn(self.d_conv_dim*2)
        self.block2 = DiscBlock(self.d_conv_dim*2, self.d_conv_dim*4)
        self.block3 = DiscBlock(self.d_conv_dim*4, self.d_conv_dim*4)
        self.block4 = DiscBlock(self.d_conv_dim*4, self.d_conv_dim*8)
        self.block5 = DiscBlock(self.d_conv_dim*8, self.d_conv_dim*8)
        self.relu = nn.ReLU(inplace=True)
        self.snlinear1 = snlinear(in_features=self.d_conv_dim*8, out_features=1)
        self.sn_embedding1 = sn_embedding(args.num_classes, self.d_conv_dim*8)

        # Weight init
        self.apply(init_weights)
        nn.init.xavier_uniform_(self.sn_embedding1.weight)

    def forward(self, x, labels):
        # n x 3 x 64 x 64
        h0 = self.opt_block1(x) # n x d_conv_dim   x 32 x 32
        h1 = self.block1(h0)    # n x d_conv_dim*2 x 16 x 16
        h1 = self.self_attn(h1) # n x d_conv_dim*2 x 16 x 16
        h2 = self.block2(h1)    # n x d_conv_dim*4 x 8 x 8
        h3 = self.block3(h2)    # n x d_conv_dim*4 x  4 x  4
        h4 = self.block4(h3)    # n x d_conv_dim*8 x 2 x  2
        h5 = self.block5(h4, downsample=False)  # n x d_conv_dim*8 x 2 x 2
        h5 = self.relu(h5)              # n x d_conv_dim*8 x 2 x 2
        h6 = torch.sum(h5, dim=[2,3])   # n x d_conv_dim*8
        output1 = torch.squeeze(self.snlinear1(h6)) # n x 1
        # Projection
        h_labels = self.sn_embedding1(labels)   # n x d_conv_dim*8
        proj = torch.mul(h6, h_labels)          # n x d_conv_dim*8
        output2 = torch.sum(proj, dim=[1])      # n x 1
        # Out
        output = output1 + output2              # n x 1
        return output    