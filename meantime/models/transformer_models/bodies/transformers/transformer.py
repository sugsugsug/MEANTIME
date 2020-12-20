from meantime.models.transformer_models.utils import PositionwiseFeedForward, SublayerConnection

import torch
from torch import nn as nn
from torch.nn import functional as F
import time
from meantime.models.transformer_models.utils.fake_quant import QuantizedLinear


class TransformerBlock(nn.Module):
    def __init__(self, args):
        super().__init__()
        attn_heads = args.num_heads
        hidden = args.hidden_units
        feed_forward_hidden = 4 * hidden
        dropout = args.dropout
        self.attention = MultiHeadedAttention(h=attn_heads, d_model=hidden, dropout=dropout)
        self.feed_forward = PositionwiseFeedForward(d_model=hidden, d_ff=feed_forward_hidden, dropout=dropout, act='gelu', middle_drop=False)
        self.input_sublayer = SublayerConnection(args=args, size=hidden, dropout=dropout)
        self.output_sublayer = SublayerConnection(args=args, size=hidden, dropout=dropout)
        self.dropout = nn.Dropout(p=dropout)
        self.args = args

    def forward(self, x, mask, layer, info):
        x = self.input_sublayer(x, lambda _x: self.attention(_x, _x, _x, mask=mask, layer=layer, info=info))
        t1 = time.time()
        x = self.output_sublayer(x, self.feed_forward)
        print('linear ', time.time() - t1)
        return x


class MultiHeadedAttention(nn.Module):
    """
    Take in model size and number of heads.
    """
    
    def __init__(self, h, d_model, dropout=0.1):
        super().__init__()
        assert d_model % h == 0

        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.scale = 1 / (self.d_k ** 0.5)

        ### linear -> 
        #self.linear_layers = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(3)])
        self.output_linear = nn.Linear(d_model, d_model)
        self.linear_layers = nn.ModuleList([QuantizedLinear(d_model,d_model, mode='ema', weight_bits=8) for _ in range(3)])
        #self.output_linear = QuantizedLinear(d_model, d_model, mode='ema', weight_bits=8)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None, layer=None, info=None):
        batch_size = query.size(0)

        t1 = time.time()
        # 1) Do all the linear projections in batch from d_model => h x d_k
        if info is not None:
            info['input_seq' + str(layer)] = value
        query, key, value = [l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linear_layers, (query, key, value))]
        if info is not None:
            info['value_seq' + str(layer)] = value

        # 2) Apply attention on all the projected vectors in batch.
        x, attn, t2 = self.attention(query, key, value, mask=mask, layer=layer, info=info)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)
        if info is not None:
            info['attn_seq' + str(layer)] = x

        t3 = time.time()
        x =  self.output_linear(x)
        if info is not None:
            info['output_seq' + str(layer)] = x

        t4 = time.time()
        #print('first, we can ',t2-t1)
        #print('second, we can\'t ',t3-t2)
        #print('third, we can ',t4-t3)
        return x

    def attention(self, query, key, value, mask=None, layer=None, info=None):
        scores = torch.matmul(query, key.transpose(-2, -1))
        t2 = time.time()

        scores = scores * self.scale

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        p_attn = F.softmax(scores, dim=-1)
        if info is not None:
            info['attn_scores' + str(layer)] = p_attn

        p_attn = self.dropout(p_attn)
        return torch.matmul(p_attn, value), p_attn, t2