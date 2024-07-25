import torch
from einops import repeat
from torch import nn
# from module import Attention, PreNorm, FeedForward
from models.cnn2d_transformer.temporal_transformer.module import Attention, PreNorm, FeedForward
import numpy as np


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.norm = nn.LayerNorm(dim)
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)


class TemporalTransformer(nn.Module):
    def __init__(self, dim = 512, depth = 6, heads = 8, dim_head = 64, dropout = 0., scale_dim = 4, ):
        super().__init__()

        self.temporal_token = nn.Parameter(torch.randn(1, 1, dim))
        self.temporal_transformer = Transformer(dim, depth, heads, dim_head, dim*scale_dim, dropout)


    def forward(self, x):
        """
        Input: (batch_size, num_segments, dim)
        """
        b, n, d = x.size()
        cls_temporal_tokens = repeat(self.temporal_token, '() n d -> b n d', b=b)
        x = torch.cat((cls_temporal_tokens, x), dim=1)
        x = self.temporal_transformer(x)

        return x[:, 0], x[:, 1:]
    

if __name__ == "__main__":
    feature = torch.randn(4, 50, 512).cuda()
    model = TemporalTransformer().cuda()
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    parameters = sum([np.prod(p.size()) for p in parameters]) / 1_000_000
    print('Trainable Parameters: %.3fM' % parameters)
    out = model(feature)
    print("Shape of out :", out.shape)      # [B, num_classes]

    