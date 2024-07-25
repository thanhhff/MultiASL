import timm
from torch import nn, Tensor
import math
import torch


class Query2Label(nn.Module):
    def __init__(self, video_encoder, num_classes):
        super().__init__()
        hidden_dim = 512
        self.backbone = timm.create_model(video_encoder, pretrained=True, num_classes=hidden_dim)
        self.pos_encod = PositionalEncoding(d_model=hidden_dim)
        self.transformer = nn.Transformer(d_model=hidden_dim, batch_first=True)
        self.query_embed = nn.Embedding(num_classes, hidden_dim)
        self.fc = GroupWiseLinear(num_classes, hidden_dim, bias=True)

    def forward(self, images, is_training=True):
        b, d, t, c, h, w = images.size()
        images = images.reshape(-1, c, h, w)
        x = self.backbone(images)
        x = x.reshape(b, d, t, -1)

        x = torch.max(x, dim=1)[0]

        x = self.pos_encod(x)
        query_embed = self.query_embed.weight.unsqueeze(0).repeat(b, 1, 1)
        hs = self.transformer(x, query_embed) # b, t, d
        out = self.fc(hs)

        # apply softmax 
        out = torch.log_softmax(out, dim=1)
        return out
    


class PositionalEncoding(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 50):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe, persistent=False)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        x = x + self.pe[:, :x.size(1)]
        out = self.dropout(x)
        return out


class GroupWiseLinear(nn.Module):
    # could be changed to: 
    # output = torch.einsum('ijk,zjk->ij', x, self.W)
    # or output = torch.einsum('ijk,jk->ij', x, self.W[0])
    def __init__(self, num_class, hidden_dim, bias=True):
        super().__init__()
        self.num_class = num_class
        self.hidden_dim = hidden_dim
        self.bias = bias

        self.W = nn.Parameter(torch.Tensor(1, num_class, hidden_dim))
        if bias:
            self.b = nn.Parameter(torch.Tensor(1, num_class))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.W.size(2))
        for i in range(self.num_class):
            self.W[0][i].data.uniform_(-stdv, stdv)
        if self.bias:
            for i in range(self.num_class):
                self.b[0][i].data.uniform_(-stdv, stdv)

    def forward(self, x):
        # x: B,K,d
        x = (self.W * x).sum(-1)
        if self.bias:
            x = x + self.b
        return x