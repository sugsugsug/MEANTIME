import torch.nn as nn
from .gelu import GELU
from meantime.models.transformer_models.utils.fake_quant import QuantizedLinear


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1, act='gelu', middle_drop=True):
        super(PositionwiseFeedForward, self).__init__()
        ### linear -> 
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        #self.w_1 = QuantizedLinear(d_model, d_ff, mode='ema', weight_bits=8)
        #self.w_2 = QuantizedLinear(d_ff, d_model, mode='ema', weight_bits=8)

        self.dropout = nn.Dropout(dropout)
        if act == 'gelu':
            self.activation = GELU()
        elif act == 'relu':
            self.activation = nn.ReLU()
        else:
            raise ValueError
        self.middle_drop = middle_drop

    def forward(self, x):
        # return self.w_2(self.dropout(self.activation(self.w_1(x))))
        if self.middle_drop:
            return self.w_2(self.dropout(self.activation(self.w_1(x))))
        else:
            return self.w_2(self.activation(self.w_1(x)))
