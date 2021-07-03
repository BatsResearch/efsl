import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import models
import utils
from .models import register
from .masking_utils import encoder_wrapper


@register('cosine')
class Cosine(nn.Module):

    def __init__(self, encoder, encoder_args={}, method='cos', temp=10., temp_learnable=True, **kwargs):
        super().__init__()

        self.encoder = models.make(encoder, **encoder_args)
        self.encoder_name = encoder

        self.method = method
        if temp_learnable:
            self.temp = nn.Parameter(torch.tensor(temp))
        else:
            self.temp = temp

    def forward(self, x_shot, x_pseudo, x_query, return_log=False):
        """
        param x_shot: num episodes x N classes x k shot(s) x 3 channels x 84 pixels x 84 pixels
        param x_query: num episodes x Nq x 3 channels x 84 pixels x 84 pixels
        param x_pseudo: num episodes x N classes x p shot(s) x 3 channels x 84 pixels x 84 pixels
        """

        x_shot, x_pseudo, x_query = encoder_wrapper(self.encoder_name, self.encoder, x_shot, x_pseudo, x_query)

        x_query = x_query.view(*x_query.shape[:2], -1)

        proto = torch.cat([x_shot, x_pseudo], dim=2).mean(2).view(*x_shot.shape[:2], -1)

        logits = utils.distance(proto=proto, x_query=x_query, method=self.method, temp=self.temp)
        return logits, list()
