from __future__ import absolute_import, division, print_function, unicode_literals
from enum import Enum, auto
import logging
from abc import ABC, abstractmethod

import torch 
import numpy as np 
import torch.nn as nn 
import torch.nn.functional as F
from torch.nn.parameter import Parameter

def get_dynamic_scale(x, bits):
    with torch.set_grad_enabled(False):
        threshold = x.abs().max()
    return get_scale(bits, threshold)

def get_scale(bits, threshold):
    return calc_max_quant_value(bits) / threshold

def calc_max_quant_value(bits):
    return 2 ** (bits - 1) - 1

def quantize(input, scale, bits):
    thresh = calc_max_quant_value(bits)
    return input.mul(scale).round().clamp(-thresh, thresh)

def dequantize(input, scale):
    return input.div(scale)

class FakeLinearQuantization(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, scale, bits=8):
        return dequantize(quantize(input, scale, bits), scale)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None

_fake_quantize = FakeLinearQuantization.apply

class QuantizationMode(Enum):
    NONE = auto()
    DYNAMIC = auto()
    EMA = auto()

class QuantizedLayer(ABC):
    """Quantized Layer interface"""

    def __init__(self, *args, weight_bits=8, mode="none", **kwargs):
        if weight_bits < 2:
            raise ValueError(f"weight_bits={weight_bits} must be higher than 1 ")
        super().__init__(*args, **kwargs)
        self.weight_bits = weight_bits
        self.mode = QuantizationMode[mode.upper()]
        self.register_buffer("_step", torch.zeros(1))
        # buffers for inference
        self.register_buffer("quantized_weight", None)
        self.register_buffer("_weight_scale", None)

    def forward(self, input):
        if self.mode == QuantizationMode.NONE:
            return super().forward(input)
        if self.training:
            out = self.training_quantized_forward(input)
            self._step += 1
        else:
            out = self.inference_quantized_forward(input)
        return out

    @abstractmethod
    def training_quantized_forward(self, input):
        """Implement forward method to be used while training"""

    @abstractmethod
    def inference_quantized_forward(self, input):
        """Implement forward method to be used while evaluating"""

    @property
    def fake_quantized_weight(self):
        return _fake_quantize(self.weight, self.weight_scale, self.weight_bits)

    @property
    def weight_scale(self):
        return (
            get_dynamic_scale(self.weight, self.weight_bits)
            if self.training
            else self._weight_scale
        )

    def train(self, mode=True):
        # If training is not equal to mode, call func
        if self.training != mode:
            if mode:
                self._train()
            else:
                self._eval()
        super().train(mode)

    def _train(self):
        pass

    def _eval(self):
        self._weight_scale = self.weight_scale
        self.quantized_weight = quantize(self.weight, self.weight_scale, self.weight_bits)

class QuantizedLinear(QuantizedLayer, nn.Linear):
    """Linear layer with quantization aware training"""

    def __init__(
        self, *args, activation_bits=8, requantize_output=True, ema_decay=0.9999, **kwargs
    ):
        super().__init__(*args, **kwargs)
        if activation_bits < 2:
            raise ValueError(f"activation_bits={activation_bits} must be higher than 1 ")
        self.activation_bits = activation_bits
        self.accumulation_bits = 32
        self.ema_decay = ema_decay
        self.requantize_output = requantize_output
        self.register_buffer("input_thresh", torch.zeros(1))
        if self.requantize_output:
            self.register_buffer("output_thresh", torch.zeros(1))
        # bias is 32 bits int and accumulated to 32.
        if kwargs.get("bias", True):
            self.register_buffer("_quantized_bias", None)
            self.register_buffer("bias_scale", None)

    def training_quantized_forward(self, input):
        assert self.training, "should only be called when training"
        if self.mode == QuantizationMode.EMA:
            self._update_ema(self.input_thresh, input.detach())
        input_scale = self._get_input_scale(input)
        # fake_quantized(x) is dequant(quant(x))
        # so, we do quantization aware training here with 32f
        out = F.linear(
            _fake_quantize(input, input_scale, self.activation_bits),
            self.fake_quantized_weight,
            # bias is untouched while training
            self.bias,
        )
        if self.requantize_output:
            if self.mode == QuantizationMode.EMA:
                self._update_ema(self.output_thresh, out.detach())
            out = _fake_quantize(out, self._get_output_scale(out), self.activation_bits)
        return out

    def inference_quantized_forward(self, input):
        """Simulate quantized inference. quantize input and perform calculation with only integer numbers.
        This function should only be used while doing inference"""
        assert not self.training, "should only be called when not training"
        input_scale = self._get_input_scale(input)
        self.bias_scale = self.weight_scale * input_scale
        quantized_input = quantize(input, input_scale, self.activation_bits)
        # Do integer matmul.
        out = F.linear(quantized_input, self.quantized_weight, self.quantized_bias)
        # Bias scale is input_scale * weight_scale.
        out = dequantize(out, self.bias_scale)
        # we can save activation mem
        if self.requantize_output:
            output_scale = self._get_output_scale(out)
            out = dequantize(quantize(out, output_scale, self.activation_bits), output_scale)
        return out

    def _eval(self):
        super()._eval()
        if self.mode == QuantizationMode.EMA and self.bias is not None:
            self.bias_scale = self._get_input_scale() * self.weight_scale
            # bias is quantized to int32
            self.quantized_bias = quantize(self.bias, self.bias_scale, self.accumulation_bits)

    @property
    def quantized_bias(self):
        try:
            if self.mode == QuantizationMode.EMA:
                bias = self._quantized_bias
            else:
                raise RuntimeError(f"Unknown quantization mode: {self.mode}")
        except AttributeError:
            bias = None
        return bias

    @quantized_bias.setter
    def quantized_bias(self, value):
        self._quantized_bias = value

    def _get_input_scale(self, input=None):
        return self._get_activation_scale(input, self.input_thresh)

    def _get_output_scale(self, output=None):
        return self._get_activation_scale(output, self.output_thresh)

    def _get_activation_scale(self, activation, threshold):
        if self.mode == QuantizationMode.EMA:
            scale = get_scale(self.activation_bits, threshold)
        return scale

    def _update_ema(self, ema, input, reduce_fn=lambda x: x.abs().max()):
        """Update exponential moving average (EMA) of activations thresholds.
        the reduce_fn calculates the current threshold from the input tensor"""
        if self._step == 0:
            ema.fill_(reduce_fn(input))
        else:
            ema.sub_((1 - self.ema_decay) * (ema - reduce_fn(input)))


class QuantizedEmbedding(QuantizedLayer, nn.Embedding):
    """Embedding layer with quantization aware training"""

    def training_quantized_forward(self, input):
        """Return quantized embeddings"""
        assert self.training, "should only be called when training"
        return F.embedding(
            input,
            # float32
            self.fake_quantized_weight,
            self.padding_idx,
            self.max_norm,
            self.norm_type,
            self.scale_grad_by_freq,
            self.sparse,
        )

    def inference_quantized_forward(self, input):
        """forward to be used during inference"""
        assert not self.training, "should only be called when not training"
        q_embeddings = F.embedding(
            input,
            # int8
            self.quantized_weight,
            self.padding_idx,
            self.max_norm,
            self.norm_type,
            self.scale_grad_by_freq,
            self.sparse,
        )
        # on the fly dequantizing
        return dequantize(q_embeddings, self.weight_scale)