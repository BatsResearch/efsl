import torch
import torch.nn as nn
from .models import register
from .masking_utils import encoder_wrapper
import models
from .masking_utils import MultiBlock


@register('relation-net')
class RelationNet(nn.Module):

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

        hdim = y[0]

        self.conv = MultiBlock(inplanes=hdim * 2, channels=[hdim], max_pool=False)
        self.linear = nn.Sequential(
            nn.Linear(hdim, 300),
            nn.LeakyReLU(0.1),
            nn.Linear(300, 1),
            nn.Sigmoid()
        )

    def forward(self, x_shot, x_pseudo, x_query, return_log=False):

        x_shot, x_pseudo, x_query = encoder_wrapper(self.encoder_name, self.encoder, x_shot, x_pseudo, x_query)
        protos = torch.cat([x_shot, x_pseudo], dim=2).mean(dim=2).unsqueeze(dim=1)
        x_query = x_query.unsqueeze(dim=2)
        expanded_query = torch.cat([x_query.expand([-1, -1, x_shot.shape[1], -1, -1, -1]), protos.expand([-1, x_query.shape[1], -1, -1, -1, -1])], dim=3)
        structure_shape = list(expanded_query.shape[:3])
        feature_shape = list(expanded_query.shape[3:])

        flat_query = expanded_query.view([-1] + feature_shape)

        h = self.conv(flat_query)
        h = h.mean(dim=(-1, -2))
        h = self.linear(h)
        logits = h.view(structure_shape)

        return logits, list()