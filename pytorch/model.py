"""LadaGAN model for Pytorch.

Reference:
  - [Efficient generative adversarial networks using linear 
    additive-attention Transformers](https://arxiv.org/abs/2401.09596)
"""
import torch
from torch import nn
from einops import rearrange, reduce
from torch import nn, einsum


# G BLOCKS

def pixel_upsample(x, H, W):
    B, N, C = x.size()
    assert N == H*W
    x = x.permute(0, 2, 1)
    x = x.view(-1, C, H, W)
    x = nn.PixelShuffle(2)(x)
    B, C, H, W = x.size()
    return x, H, W, C

class AdditiveAttention(nn.Module):
    def __init__(self, dim, heads=8):
        super().__init__()
        self.dim_head = dim // heads
        self.heads = heads
        self.scale = self.dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, dim * 3)

        self.q_attn = nn.Linear(dim, heads)  # Apply q_attn before rearrange
        self.to_out = nn.Linear(dim, dim)

    def forward(self, x):
        n, device, h = x.shape[1], x.device, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        
        q, k, v = qkv  

        q_attn_logits = self.q_attn(q) * self.scale  # (b, n, h)
        q_attn_logits = rearrange(q_attn_logits, 'b n h -> b h n')  
        q_attn = q_attn_logits.softmax(dim=-1) 

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), (q, k, v))

        # Compute global_q
        global_q = einsum('b h n, b h n d -> b h d', q_attn, q)  
        global_q = rearrange(global_q, 'b h d -> b h () d')

        k = k * global_q  
        u = v * k
        r = rearrange(u, 'b h n d -> b n (h d)')

        return self.to_out(r)

class SelfModulatedLayerNorm(nn.Module):
    def __init__(self, dim, cond_dim):
        super().__init__()
        self.param_free_norm = nn.LayerNorm(dim, eps=1e-6, elementwise_affine=False)
        self.h =  nn.Sequential(
            nn.Linear(cond_dim, dim),
            nn.ReLU()
        )
        self.mlp_gamma = nn.Linear(dim, dim)
        self.mlp_beta = nn.Linear(dim, dim)

    def forward(self, inputs):
        x, cond_input = inputs
        bs = x.shape[0]
        cond_input = cond_input.reshape((bs, -1))
        cond_input = self.h(cond_input)

        gamma = self.mlp_gamma(cond_input)
        gamma = gamma.reshape((bs, 1, -1))
        beta = self.mlp_beta(cond_input)
        beta = beta.reshape((bs, 1, -1))

        out = self.param_free_norm(x)
        out = out * gamma + beta

        return out

class SMLadaformer(nn.Module):
    def __init__(self, dim, cond_dim, heads=4, mlp_dim=512):
        super().__init__()
        self.ln_1 = SelfModulatedLayerNorm(dim, cond_dim)
        self.attn = AdditiveAttention(dim, heads=heads)
        
        self.ln_2 = SelfModulatedLayerNorm(dim, cond_dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, dim),
        )

    def forward(self, inp):
        x, z = inp
        x = self.attn(self.ln_1([x, z])) + x
        return self.mlp(self.ln_2([x, z])) 

class Generator(nn.Module):
    def __init__(self, img_size=64, dim=[1024, 256, 64], noise_dim=128, 
                heads=[4, 4, 4], mlp_dim=[512, 512, 512]):
        super(Generator, self).__init__()
        self.init = nn.Sequential(
            nn.Linear(noise_dim, 64 * dim[0], bias=False),
        )             
        self.block_64 = SMLadaformer(dim[0], noise_dim, heads[0], mlp_dim[0])
        self.pos_64 = nn.Parameter(torch.randn(1, 64, dim[0]))
        self.conv_256 = nn.Conv2d(dim[1], dim[1], 3, 1, 1)
        self.block_256 = SMLadaformer(dim[1], noise_dim, heads[1], mlp_dim[1])
        self.pos_256 = nn.Parameter(torch.randn(1, 256, dim[1]))
        self.conv_1024 = nn.Conv2d(dim[2], dim[2], 3, 1, 1)
        self.block_1024 = SMLadaformer(dim[2], noise_dim, heads[2], mlp_dim[2])
        self.pos_1024 = nn.Parameter(torch.randn(1, 1024, dim[2]))
        self.patch_size = img_size // 32
        self.ch_conv = nn.Conv2d(dim[2] // 4, 3, 3, 1, 1)

    def forward(self, z):
        B = z.shape[0]
        x = self.init(z)
        x = torch.reshape(x, (B, 64, -1))  
        x += self.pos_64
        x = self.block_64([x, z]) 
        x, H, W, C= pixel_upsample(x, 8, 8)
        x = self.conv_256(x)
        x = x.view(-1, C, H*W)
        x = x.permute(0, 2, 1)
        
        x += self.pos_256
        x = self.block_256([x, z]) 
        x, H, W, C = pixel_upsample(x, H, W)
        x = self.conv_1024(x)
        x = x.view(-1, C, H*W)
        x = x.permute(0, 2, 1)
        x += self.pos_1024
        x = self.block_1024([x, z]).permute(0, 2, 1).reshape([B, -1, 32, 32])
        if self.patch_size != 1:
            x = nn.PixelShuffle(self.patch_size)(x)
        img = self.ch_conv(x)
        return img

# D BLOCKS        

class DownBlock(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(DownBlock, self).__init__()

        self.main = nn.Sequential(
            nn.Conv2d(in_planes, out_planes, 3, 2, 1, bias=False),
            nn.BatchNorm2d(out_planes), 
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_planes, out_planes, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_planes), 
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.direct = nn.Sequential(
            nn.AvgPool2d(2, 2),
            nn.Conv2d(in_planes, out_planes, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_planes), nn.LeakyReLU(0.2, inplace=True))

    def forward(self, feat):
        return (self.main(feat) + self.direct(feat)) / 2

class Ladaformer(nn.Module):
    def __init__(self, seq_len, dim, heads=4, mlp_dim=512):
        super().__init__()
        self.ln_1 = nn.LayerNorm(dim)
        self.attn = AdditiveAttention(dim, heads=heads)
        
        self.ln_2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, dim),
        )

    def forward(self, x):
        x = self.attn(self.ln_1(x)) + x
        return self.mlp(self.ln_2(x)) + x

class Discriminator(nn.Module):
    def __init__(self, enc_dim=[64, 128, 256], out_dim=[512, 1024], heads=4, mlp_dim=512):
        super(Discriminator, self).__init__()
        self.inp_conv = nn.Sequential(
            nn.Conv2d(3, enc_dim[0], 3, 1, 1, bias=False),
             nn.LeakyReLU(0.2),
        )
        self.encoder = nn.ModuleList([
            DownBlock(enc_dim[i], enc_dim[i+1]) for i in range(len(enc_dim)-1)
        ])
        self.pos_256 = nn.Parameter(torch.randn(1, 256, enc_dim[2]))
        self.block_256 = Ladaformer(256, enc_dim[2], heads, mlp_dim)
        self.conv_256 = nn.Conv2d(enc_dim[2] * 4, out_dim[0], 3, 1, 1)

        self.logits = nn.Sequential(
            nn.Conv2d(out_dim[0], out_dim[1], 1, 1, 0, bias=False),
            nn.LeakyReLU(0.2),
            nn.Conv2d(out_dim[1], 1, 4, 1, 0, bias=False),
        )
        
    def forward(self, x):
        x = self.inp_conv(x)
        for down in self.encoder:
            x = down(x)
        B, C, H, W = x.shape
        x = x.reshape([B, C, H * W]).permute([0, 2, 1])
        x += self.pos_256
        x = self.block_256(x) 
        x = x.permute([0, 2, 1]).reshape([B, C, H, W])
        x = nn.PixelUnshuffle(2)(x)
        x = self.conv_256(x)
        x = self.logits(x).view(B, -1)
        return x
