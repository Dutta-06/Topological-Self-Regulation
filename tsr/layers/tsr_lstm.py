import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from tsr.layers.tsr_linear import ACTIVATION_FNS, ACTIVATION_NAMES, NUM_ACTIVATIONS

class TSRLSTMCell(nn.Module):
    """LSTM Cell with topological self-regulation and activation mixing.
    """
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        bias: bool = True,
        gate_init: float = 3.0,
        act_init: str = "relu",
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # LSTM weights: 4 * hidden_size chunks (i, f, g, o)
        self.weight_ih = nn.Parameter(torch.empty(4 * hidden_size, input_size))
        self.weight_hh = nn.Parameter(torch.empty(4 * hidden_size, hidden_size))
        
        if bias:
            self.bias_ih = nn.Parameter(torch.empty(4 * hidden_size))
            self.bias_hh = nn.Parameter(torch.empty(4 * hidden_size))
        else:
            self.register_parameter('bias_ih', None)
            self.register_parameter('bias_hh', None)
            
        # TSR specific parameters
        self.tsr_gate = nn.Parameter(torch.full((hidden_size,), gate_init))
        
        act_idx = ACTIVATION_NAMES.index(act_init) if act_init in ACTIVATION_NAMES else 0
        act_logits = torch.zeros(NUM_ACTIVATIONS)
        act_logits[act_idx] = 3.0
        self.act_weights = nn.Parameter(act_logits)
        
        self.reset_parameters()
        
    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size) if self.hidden_size > 0 else 0
        for weight in self.parameters():
            if weight is not self.tsr_gate and weight is not self.act_weights:
                nn.init.uniform_(weight, -stdv, stdv)
                
    def forward(self, x: torch.Tensor, hx: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = x.size(0)
        
        if hx is None:
            h_t = torch.zeros(batch_size, self.hidden_size, dtype=x.dtype, device=x.device)
            c_t = torch.zeros(batch_size, self.hidden_size, dtype=x.dtype, device=x.device)
        else:
            h_t, c_t = hx
            
        gates = F.linear(x, self.weight_ih, self.bias_ih) + F.linear(h_t, self.weight_hh, self.bias_hh)
        
        # Chunk into i, f, g, o
        i_g, f_g, g_g, o_g = gates.chunk(4, 1)
        
        i_t = torch.sigmoid(i_g)
        f_t = torch.sigmoid(f_g)
        g_t = torch.tanh(g_g)
        o_t = torch.sigmoid(o_g)
        
        c_t = f_t * c_t + i_t * g_t
        
        # TSR Activation mixing on cell state instead of tanh
        act_mix = F.softmax(self.act_weights, dim=0)
        mixed_c_t = torch.zeros_like(c_t)
        for i, act_fn in enumerate(ACTIVATION_FNS):
            mixed_c_t = mixed_c_t + act_mix[i] * act_fn(c_t)
            
        # Standard LSTM output gating
        h_t_base = o_t * mixed_c_t
        
        # TSR Differentiable gating
        gate_values = torch.sigmoid(self.tsr_gate)
        h_t_final = h_t_base * gate_values
        
        return h_t_final, c_t
        
    def gate_values(self) -> torch.Tensor:
        with torch.no_grad():
            return torch.sigmoid(self.tsr_gate)

    def effective_neurons(self) -> int:
        return int((self.gate_values() > 0.5).sum().item())

    def activation_distribution(self) -> dict:
        with torch.no_grad():
            mix = F.softmax(self.act_weights, dim=0)
            return {name: mix[i].item() for i, name in enumerate(ACTIVATION_NAMES)}

    def dominant_activation(self) -> str:
        with torch.no_grad():
            idx = self.act_weights.argmax().item()
            return ACTIVATION_NAMES[idx]
            
    def prune_neurons(self, indices_to_remove: torch.Tensor) -> None:
        if len(indices_to_remove) == 0:
            return
            
        keep_mask = torch.ones(self.hidden_size, dtype=torch.bool)
        keep_mask[indices_to_remove] = False
        keep_indices = keep_mask.nonzero(as_tuple=True)[0]
        
        new_hidden_size = len(keep_indices)
        
        # To prune out chunks, we need the indices for all 4 gates
        # In chunking, indices are 0:h, h:2h, 2h:3h, 3h:4h
        keep_indices_4x = []
        for i in range(4):
            keep_indices_4x.append(keep_indices + i * self.hidden_size)
        keep_indices_4x = torch.cat(keep_indices_4x)
        
        self.weight_ih = nn.Parameter(self.weight_ih.data[keep_indices_4x])
        # weight_hh needs row AND column pruning
        w_hh = self.weight_hh.data[keep_indices_4x]
        w_hh = w_hh[:, keep_indices]
        self.weight_hh = nn.Parameter(w_hh)
        
        if getattr(self, 'bias_ih', None) is not None:
            self.bias_ih = nn.Parameter(self.bias_ih.data[keep_indices_4x])
            self.bias_hh = nn.Parameter(self.bias_hh.data[keep_indices_4x])
            
        self.tsr_gate = nn.Parameter(self.tsr_gate.data[keep_indices])
        self.hidden_size = new_hidden_size
        
    def prune_input_channels(self, indices_to_remove: torch.Tensor) -> None:
        if len(indices_to_remove) == 0:
            return
            
        keep_mask = torch.ones(self.input_size, dtype=torch.bool)
        keep_mask[indices_to_remove] = False
        keep_indices = keep_mask.nonzero(as_tuple=True)[0]
        
        self.weight_ih = nn.Parameter(self.weight_ih.data[:, keep_indices])
        self.input_size = len(keep_indices)
        
    def grow_neurons(self, n: int, init_scale: float = 0.001) -> None:
        if n <= 0:
            return
            
        device = self.weight_ih.device
        dtype = self.weight_ih.dtype
        
        # We need to insert n rows into each of the 4 gate chunks
        def grow_4x_rows(weight_data, n_cols):
            chunks = weight_data.chunk(4, dim=0)
            new_chunks = []
            for c in chunks:
                new_w = torch.randn(n, n_cols, device=device, dtype=dtype) * init_scale
                new_chunks.append(torch.cat([c, new_w], dim=0))
            return torch.cat(new_chunks, dim=0)
            
        def grow_4x_bias(bias_data):
            chunks = bias_data.chunk(4, dim=0)
            new_chunks = []
            for c in chunks:
                new_b = torch.zeros(n, device=device, dtype=dtype)
                new_chunks.append(torch.cat([c, new_b], dim=0))
            return torch.cat(new_chunks, dim=0)
            
        new_weight_ih = grow_4x_rows(self.weight_ih.data, self.input_size)
        
        # weight_hh needs row expansion (for 4 gates) AND column expansion (for new hidden_size inputs)
        # first expand columns with zeros so they don't impact existing state abruptly
        w_hh = self.weight_hh.data
        new_cols = torch.zeros(4 * self.hidden_size, n, device=device, dtype=dtype)
        w_hh = torch.cat([w_hh, new_cols], dim=1)
        
        # then expand rows
        new_weight_hh = grow_4x_rows(w_hh, self.hidden_size + n)
        
        self.weight_ih = nn.Parameter(new_weight_ih)
        self.weight_hh = nn.Parameter(new_weight_hh)
        
        if getattr(self, 'bias_ih', None) is not None:
            self.bias_ih = nn.Parameter(grow_4x_bias(self.bias_ih.data))
            self.bias_hh = nn.Parameter(grow_4x_bias(self.bias_hh.data))
            
        new_g = torch.full((n,), -5.0, device=device, dtype=dtype)
        self.tsr_gate = nn.Parameter(torch.cat([self.tsr_gate.data, new_g], dim=0))
        
        self.hidden_size += n
        
    def grow_input_channels(self, n: int) -> None:
        if n <= 0:
            return
            
        device = self.weight_ih.device
        dtype = self.weight_ih.dtype
        
        new_cols = torch.zeros(4 * self.hidden_size, n, device=device, dtype=dtype)
        self.weight_ih = nn.Parameter(torch.cat([self.weight_ih.data, new_cols], dim=1))
        self.input_size += n

    def extra_repr(self) -> str:
        return (
            f"input_size={self.input_size}, hidden_size={self.hidden_size}, "
            f"effective={self.effective_neurons()}, "
            f"bias={getattr(self, 'bias_ih', None) is not None}, "
            f"dominant_act={self.dominant_activation()}"
        )


class TSRLSTM(nn.Module):
    """A wrapper for TSRLSTMCell to process sequences."""
    def __init__(self, input_size: int, hidden_size: int, batch_first: bool = True):
        super().__init__()
        self.batch_first = batch_first
        self.cell = TSRLSTMCell(input_size, hidden_size)
        
    def forward(self, x: torch.Tensor, hx: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if self.batch_first:
            # (batch, seq, feature) -> (seq, batch, feature)
            x = x.transpose(0, 1)
            
        seq_len, batch_size, _ = x.size()
        
        if hx is None:
            h_t = torch.zeros(batch_size, self.cell.hidden_size, dtype=x.dtype, device=x.device)
            c_t = torch.zeros(batch_size, self.cell.hidden_size, dtype=x.dtype, device=x.device)
        else:
            h_t, c_t = hx
            
        outputs = []
        for t in range(seq_len):
            h_t, c_t = self.cell(x[t], (h_t, c_t))
            outputs.append(h_t)
            
        out = torch.stack(outputs, dim=0)
        
        if self.batch_first:
            out = out.transpose(0, 1)
            
        return out, (h_t, c_t)
