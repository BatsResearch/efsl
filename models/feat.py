import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from .models import register
from .masking_utils import encoder_wrapper
import models
import utils

@register('feat')
class Feat(nn.Module):

    def __init__(self, encoder, encoder_args, method='cos', temp=10., temp_learnable=True, **kwargs):
        super().__init__()

        self.encoder = models.make(encoder, **encoder_args)
        self.encoder_name = encoder

        self.method = method
        if temp_learnable:
            self.temp = nn.Parameter(torch.tensor(temp))
        else:
            self.temp = temp

        input_shape = kwargs.get('input_shape')
        x = torch.randn([1] + input_shape).cuda()
        embedding_f = self.encoder
        if hasattr(self.encoder, 'encoder'):
            embedding_f = self.encoder.encoder
        y = embedding_f(x).shape[1:]

        hdim = y[0] * y[1] * y[2]
        self.slf_attn = MultiHeadAttention(1, hdim, hdim, hdim, dropout=0.5)

    def forward(self, x_shot, x_pseudo, x_query, return_log=False):

        x_shot, x_pseudo, x_query = encoder_wrapper(self.encoder_name, self.encoder, x_shot, x_pseudo, x_query)
        protos = torch.cat([x_shot, x_pseudo], dim=2).mean(2).view(*x_shot.shape[:2], -1)
        x_query = x_query.view(*x_query.shape[:2], -1)
        n_batches = protos.shape[0]
        # n_way = protos.shape[1]

        protos_adapted = list()
        for i in range(n_batches):
            proto = protos[i:i + 1]
            # proto = proto.transpose(0, 1)
            proto = self.slf_attn(proto, proto, proto)
            protos_adapted.append(proto)

        protos = torch.cat(protos_adapted, dim=0)

        logits = utils.distance(proto=protos, x_query=x_query, method=self.method, temp=self.temp)
        return logits, list()


class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v):
        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.temperature
        log_attn = F.log_softmax(attn, 2)
        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = torch.bmm(attn, v)
        return output, attn, log_attn


class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_v)))

        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5))
        self.layer_norm = nn.LayerNorm(d_model)

        self.fc = nn.Linear(n_head * d_v, d_model)
        nn.init.xavier_normal_(self.fc.weight)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()

        residual = q
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k)  # (n*b) x lq x dk
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k)  # (n*b) x lk x dk
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v)  # (n*b) x lv x dv

        output, attn, log_attn = self.attention(q, k, v)

        output = output.view(n_head, sz_b, len_q, d_v)
        output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1)  # b x lq x (n*dv)

        output = self.dropout(self.fc(output))
        output = self.layer_norm(output + residual)

        return output
