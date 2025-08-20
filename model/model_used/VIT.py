import torch
from torch import nn
import torch.nn.functional as F
from  functools import partial

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

from util.register import MODEL_REGISTRY
import math

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

class decoder_linear(nn.Module):
    def __init__(self, in_c = 256, out_c = 1):
        super().__init__()
        self.linear_up = nn.Sequential(
            nn.Linear(in_c, 1024),
            nn.BatchNorm1d(1024),
            nn.GELU(),
            nn.Linear(1024, 4096),
            nn.BatchNorm1d(4096),
            nn.GELU(),
            nn.Linear(4096, 3 * 128 ** 2)
        )
    def forward(self, latent):
        linear_output = self.linear_up(latent)
        return linear_output

class decoder_linear_all(nn.Module):
    def __init__(self, in_c = 256, out_c = 1):
        super().__init__()
        self.linear_up = nn.Sequential(
            nn.Linear(in_c, 1024),
            nn.BatchNorm1d(1024),
            nn.GELU(),
            nn.Linear(1024, 128 ** 2),
            nn.BatchNorm1d(128 ** 2),
            nn.GELU(),
            nn.Linear(128 ** 2, 512 ** 2),
            nn.BatchNorm1d(512 ** 2),
            nn.GELU(),
            nn.Linear(512 ** 2, 1024 ** 2),
            nn.ReLU()
        )
    def forward(self, latent):
        linear_output = self.linear_up(latent)
        return linear_output

class conv_up(nn.Module):
    def __init__(self, in_c):
        super().__init__()
        self.conv_seq = nn.Sequential(
            nn.BatchNorm2d(in_c),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_c, in_c, 3, 1, 1),
            nn.GELU(),
        )

    def forward(self, x):
        conv_out = self.conv_seq(x)
        return conv_out

class conv_up_res(nn.Module):
    def __init__(self, in_c):
        super().__init__()
        self.bn = nn.BatchNorm2d(in_c)
        self.up = nn.Upsample(scale_factor=2)
        self.conv_seq = nn.Sequential(
            nn.Conv2d(in_c, in_c, 5, 1, 2),
            nn.GELU(),
            nn.Conv2d(in_c, in_c, 3, 1, 1),
            nn.GELU()
        )

    def forward(self, x):
        x = self.bn(x)
        up = self.up(x)
        res = up
        conv_out = self.conv_seq(up) + res

        return conv_out

class decoder_conv(nn.Module):
    def __init__(self, in_c = 3, mid_c = 16):
        super().__init__()
        self.conv_intro = nn.Conv2d(in_c, mid_c, 5, 1, 2)
        self.conv_up = conv_up_res(mid_c)
        self.conv_extro = nn.Conv2d(mid_c, 1, 3, 1, 1)

    def forward(self, x):
        intro = self.conv_intro(x)
        conv_out = self.conv_up(intro)
        conv_out = self.conv_extro(conv_out)
        return conv_out

class Embedding_exp(nn.Module):
    def __init__(self, in_channels = 2, N_freqs = 20, logscale=True, mod = 'exp'):
        super(Embedding_exp, self).__init__()
        self.N_freqs = N_freqs
        self.in_channels = in_channels
        self.funcs = [torch.sin, torch.cos]
        self.out_channels = in_channels*(len(self.funcs)*N_freqs+1)

        if logscale:
            self.freq_bands = 2**torch.linspace(0, N_freqs-1, N_freqs)
        else:
            self.freq_bands = torch.linspace(1, 2**(N_freqs-1), N_freqs)

    def forward(self, x):
        out = [x]
        for freq in self.freq_bands:
            for func in self.funcs:
                out += [func(freq*x)]
        return torch.cat(out, -1)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1024):

        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).detach()
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return x

class LeakyReluLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True, **kwargs):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.leaky_relu = partial(F.leaky_relu, inplace = True) # saves a lot of memory
        self.init_weights()
        self.dropout = kwargs['dropout']

    def init_weights(self):
        with torch.no_grad():
            nn.init.uniform_(self.linear.bias, -0.05, 0.05)
            nn.init.kaiming_uniform_(self.linear.weight, mode = 'fan_in', nonlinearity = 'leaky_relu')

    def forward(self, input):
         return self.leaky_relu(F.dropout(self.linear(input), p = self.dropout))

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.norm = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        x = self.norm(x)
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout))),
                Residual(PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout)))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) 
            x = ff(x) 
        return x
    
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x

@MODEL_REGISTRY.register('vit')
class ViT(nn.Module):
    def __init__(self, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool = 'cls', channels = 1, dim_head = 64, dropout = 0., emb_dropout = 0., latent_dim = 128):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.pos_embedding = PositionalEncoding(dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)
        self.pool = pool
        self.to_latent = nn.Identity()
        self.to_small_latent = nn.Sequential(
            nn.Linear(dim, 512),
            nn.ReLU(inplace = True),
            nn.Dropout(0.15),
            nn.Linear(512, latent_dim)
        )
        dim = latent_dim
        self.mlp_head = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(inplace = True),
            nn.Linear(dim, num_classes)
        )

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)
        x = self.pos_embedding(x)
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.dropout(x)
        x = self.transformer(x)
        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]
        latent_features = self.to_latent(x)
        latent_features = self.to_small_latent(latent_features)
        cls = self.mlp_head(latent_features)
        return cls, None 

