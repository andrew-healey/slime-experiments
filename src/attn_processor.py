# mostly copied from https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py
from typing import Callable, Optional, Union, List

import torch
import torch.nn.functional as F
from torch import nn

class CustomAttnProcessor:
    r"""
    Default processor for performing attention-related computations.
    """

    def __init__(
            self,
            arr_to_write: List[Optional[torch.Tensor]],
            arr_to_write_idx: int,
    ):
        self.arr_to_write = arr_to_write
        self.arr_to_write_idx = arr_to_write_idx

    def __call__(
        self,
        attn,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        temb: Optional[torch.FloatTensor] = None,
        scale: float = 1.0
    ) -> torch.Tensor:
        residual = hidden_states

        args = ()

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states, *args) # b,q,d

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states, *args) # b,k,d
        value = attn.to_v(encoder_hidden_states, *args) # b,k,d

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)
        
        attention_probs = attn.get_attention_scores(query, key, attention_mask)

        #
        # This is the only SLIME modification
        #
        self.arr_to_write[self.arr_to_write_idx] = attention_probs
        #
        #
        #


        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states, *args)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states