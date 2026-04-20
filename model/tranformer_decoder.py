import math
import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_

def init_weights(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_uniform_(m.weight, a=1)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    else:
        for p in m.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

def PosEncoding(feats, pos):
    if pos is not None:
        return feats + pos
    else:
        return feats

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout, out_dim=None, act='gelu'):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU() if act =='gelu' else nn.ReLU()
        if out_dim is None:
            out_dim = dim
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.drop = nn.Dropout(dropout)

    @property
    def unwrapped(self):
        return self

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        return x

class MultiHeadCrossAttention(nn.Module):
    """
    Multi-head attention module for both image and text
    """

    def __init__(self, q_dim, k_dim, embed_dim, num_heads, dropout=0.1, 
        clamp_min_for_underflow = False, clamp_max_for_overflow = False):
        super(MultiHeadCrossAttention, self).__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.q_dim = q_dim
        self.k_dim = k_dim

        assert (
                self.head_dim * self.num_heads == self.embed_dim
        ), f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`: {self.num_heads})."
        self.scale = self.head_dim ** (-0.5)
        self.dropout = nn.Dropout(dropout)

        self.q_proj = nn.Linear(self.q_dim, self.embed_dim)
        self.k_proj_p = nn.Linear(self.k_dim, self.embed_dim)
        self.k_proj_sp = nn.Linear(self.k_dim, self.embed_dim)
        self.v_proj_p = nn.Linear(self.k_dim, self.embed_dim)
        self.v_proj_sp = nn.Linear(self.k_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.q_dim)
        self.clamp_min_for_underflow = clamp_min_for_underflow
        self.clamp_max_for_overflow = clamp_max_for_overflow

        self._reset_parameters()

    def extra_repr(self):
        return 'num_heads={}'.format(self.num_heads)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def _reset_parameters(self):
        # nn.init.xavier_uniform_(self.q_proj.weight)
        # nn.init.xavier_uniform_(self.k_proj.weight)
        # nn.init.xavier_uniform_(self.v_proj.weight)
        # nn.init.xavier_uniform_(self.out_proj.weight)
        trunc_normal_(self.q_proj.weight)
        trunc_normal_(self.k_proj_p.weight)
        trunc_normal_(self.k_proj_sp.weight)
        trunc_normal_(self.v_proj_sp.weight)
        trunc_normal_(self.v_proj_p.weight)
        trunc_normal_(self.out_proj.weight)
        self.q_proj.bias.data.fill_(0)
        self.k_proj_p.bias.data.fill_(0)
        self.k_proj_sp.bias.data.fill_(0)
        self.v_proj_sp.bias.data.fill_(0)
        self.v_proj_p.bias.data.fill_(0)
        self.out_proj.bias.data.fill_(0)

    def forward(self, l, sp, p, attention_mask=None):
        bsz, sen_len, embed_dim = l.size()
        bsz, sp_len, embed_dim = sp.size()
        bsz, p_len, embed_dim = p.size()

        query_l = self.q_proj(l) * self.scale
        key_sp = self.k_proj_sp(sp) * self.scale
        key_p = self.k_proj_p(p) * self.scale
        value_sp = self.v_proj_sp(sp) * self.scale
        value_p = self.v_proj_p(p) * self.scale

        key_sp = self._shape(key_sp, -1, bsz) # [B, num_heads, src_len, head_dim]
        value_sp = self._shape(value_sp, -1, bsz) # [B, num_heads, src_len, head_dim]
        key_p = self._shape(key_p, -1, bsz) # [B, num_heads, src_len, head_dim]
        value_p = self._shape(value_p, -1, bsz) # [B, num_heads, src_len, head_dim]

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_l = self._shape(query_l, sen_len, bsz).view(*proj_shape) # [B*num_heads, tgt_len, head_dim]
        key_sp = key_sp.view(*proj_shape) # [B*num_heads, src_len, head_dim]
        value_sp = value_sp.view(*proj_shape) # [B*num_heads, src_len, head_dim]
        key_p = key_p.view(*proj_shape) # [B*num_heads, src_len, head_dim]
        value_p = value_p.view(*proj_shape) # [B*num_heads, src_len, head_dim]

        # src_len = key_states.size(1)
        attn_weights_l2sp = torch.bmm(query_l, key_sp.transpose(1, 2)) # [B*num_heads, sen_len, sp_len]
        attn_weights_l2p = torch.bmm(query_l, key_p.transpose(1, 2)) # [B*num_heads, sen_len, p_len]

        if self.clamp_min_for_underflow:
            attn_weights_l2sp = torch.clamp(attn_weights_l2sp, min=-50000) # Do not increase -50000, data type half has quite limited range
            attn_weights_l2p = torch.clamp(attn_weights_l2p, min=-50000) # Do not increase -50000, data type half has quite limited range
        if self.clamp_max_for_overflow:
            attn_weights_l2sp = torch.clamp(attn_weights_l2sp, max=50000) # Do not increase 50000, data type half has quite limited range
            attn_weights_l2p = torch.clamp(attn_weights_l2p, max=50000) # Do not increase 50000, data type half has quite limited range

        attn_weights_l2sp = nn.functional.softmax(attn_weights_l2sp, dim=1)
        attn_weights_l2p = nn.functional.softmax(attn_weights_l2p, dim=1)

        attn_weights_sp2p = torch.bmm(attn_weights_l2sp.transpose(1,2), attn_weights_l2p).transpose(1,2)
        attn_weights_sp2p = nn.functional.softmax(attn_weights_sp2p, dim=-1)

        attn_probs = self.dropout(attn_weights_sp2p)
        attn_output = torch.bmm(attn_probs, value_sp)
        if attn_output.size() != (bsz * self.num_heads, p_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, p_len, self.head_dim)}, but is {attn_output.size()}"
            )

        attn_output = attn_output.view(bsz, self.num_heads, p_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, p_len, self.embed_dim)

        attn_output = self.out_proj(attn_output)

        return attn_output


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention module for both image and text
    """

    def __init__(self, q_dim, k_dim, embed_dim, num_heads, dropout=0.1, 
        clamp_min_for_underflow = False, clamp_max_for_overflow = False):
        super(MultiHeadAttention, self).__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.q_dim = q_dim
        self.k_dim = k_dim

        assert (
                self.head_dim * self.num_heads == self.embed_dim
        ), f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`: {self.num_heads})."
        self.scale = self.head_dim ** (-0.5)
        self.dropout = nn.Dropout(dropout)

        self.q_proj = nn.Linear(self.q_dim, self.embed_dim)
        self.k_proj = nn.Linear(self.k_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.k_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.q_dim)
        self.clamp_min_for_underflow = clamp_min_for_underflow
        self.clamp_max_for_overflow = clamp_max_for_overflow

        self._reset_parameters()

    def extra_repr(self):
        return 'num_heads={}'.format(self.num_heads)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def _reset_parameters(self):
        # nn.init.xavier_uniform_(self.q_proj.weight)
        # nn.init.xavier_uniform_(self.k_proj.weight)
        # nn.init.xavier_uniform_(self.v_proj.weight)
        # nn.init.xavier_uniform_(self.out_proj.weight)
        trunc_normal_(self.q_proj.weight)
        trunc_normal_(self.k_proj.weight)
        trunc_normal_(self.v_proj.weight)
        trunc_normal_(self.out_proj.weight)
        self.q_proj.bias.data.fill_(0)
        self.k_proj.bias.data.fill_(0)
        self.v_proj.bias.data.fill_(0)
        self.out_proj.bias.data.fill_(0)

    def forward(self, q, k, v, attention_mask=None):
        bsz, tgt_len, embed_dim = q.size()

        query_states = self.q_proj(q) * self.scale
        key_states = self._shape(self.k_proj(k), -1, bsz) # [B, num_heads, src_len, head_dim]
        value_states = self._shape(self.v_proj(v), -1, bsz) # [B, num_heads, src_len, head_dim]

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape) # [B*num_heads, tgt_len, head_dim]
        key_states = key_states.view(*proj_shape) # [B*num_heads, src_len, head_dim]
        value_states = value_states.view(*proj_shape) # [B*num_heads, src_len, head_dim]

        src_len = key_states.size(1)
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2)) # [B*num_heads, tgt_len, src_len]

        if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is {attn_weights.size()}"
            )

        if self.clamp_min_for_underflow:
            attn_weights = torch.clamp(attn_weights, min=-50000) # Do not increase -50000, data type half has quite limited range
        if self.clamp_max_for_overflow:
            attn_weights = torch.clamp(attn_weights, max=50000) # Do not increase 50000, data type half has quite limited range

        if attention_mask is not None:
            # [bsz, src_len]
            assert (attention_mask.dim() == 2)
            
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(1) # [B, 1, 1, src_len]
            attention_mask = attention_mask.expand(bsz, 1, tgt_len, src_len)
            
            if attention_mask.size() != (bsz, 1, tgt_len, src_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}"
                )
            # attention_mask = attention_mask.masked_fill(attention_mask == 0, float("-inf"))
            # attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights.masked_fill(attention_mask == 0, float("-inf"))
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        attn_probs = self.dropout(attn_weights)

        attn_output = torch.bmm(attn_probs, value_states)

        if attn_output.size() != (bsz * self.num_heads, tgt_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is {attn_output.size()}"
            )

        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)

        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
