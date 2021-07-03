import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import models
import utils
from .models import register
from .masking_utils import encoder_wrapper

@register('masking-model')
class MaskingModel(nn.Module):

	def __init__(self, encoder, encoder_args, masking, masking_args):
		super().__init__()

		self.encoder = models.make(encoder, **encoder_args)
		self.encoder_name = encoder
		masking_args['inplanes'] = self.encoder.out_dim * 2
		self.masking_model = models.make(masking, **masking_args)

	def forward(self, x_shot, x_pseudo, x_query):
		"""
		param x_shot: num episodes x N classes x k shot(s) x 3 channels x 84 pixels x 84 pixels
		param x_query: num episodes x Nq x 3 channels x 84 pixels x 84 pixels
		param x_pseudo: num episodes x N classes x p shot(s) x 3 channels x 84 pixels x 84 pixels
		"""

		x_shot, x_pseudo, x_query = encoder_wrapper(self.encoder_name, self.encoder, x_shot, x_pseudo, x_query)

		ep_per_batch = x_shot.shape[0]
		n_way = x_shot.shape[1]
		n_shot = x_shot.shape[2]
		n_pseudo = x_pseudo.shape[2]
		n_query = x_query.shape[1]

		a_shot = torch.mean(x_shot, dim=-4)
		a_pseudo = torch.mean(x_pseudo, dim=-4)

		total = torch.cat((a_shot, a_pseudo), dim=-3) # [2, 5, 1280, 5, 5]

		batch_shape = total.shape[:2]
		feat_shape = total.shape[2:]
		total = total.view(-1, *feat_shape) # [10, 1280, 5, 5]
		mask = self.masking_model(total)
		mask = mask.view(*batch_shape, *mask.shape[1:]).unsqueeze(dim=2)

		x_pseudo = torch.mul(x_pseudo, mask) # [ep_per_batch, n_way, n_pseudo, 640, 5, 5]

		img_shape = x_shot.shape[-3:]
		# x_shot = x_shot.view(-1, *img_shape) # shape is [10, 640, 5, 5] = [2, 5, 1, 640, 5, 5]
		# x_pseudo = x_pseudo.view(-1, *img_shape) # shape is [150, 640, 5, 5] = [2, 5, 15, 640, 5, 5]
		# x_query = x_query.view(-1, *img_shape) # shape is [150, 640, 5, 5] = [2, 5, 15, 640, 5, 5]
		#
		# x_tot = self.final(torch.cat([x_shot, x_pseudo, x_query], dim=0)) # shape is [310, 640, 5, 5] = [2, 5, 31, 640, 5, 5]
		# x_shot, x_pseudo, x_query = x_tot[:len(x_shot)], x_tot[len(x_shot):len(x_shot) + len(x_pseudo)], x_tot[len(
		# 	x_shot) + len(x_pseudo):]

		x_shot = x_shot.view(ep_per_batch, n_way, n_shot, *img_shape)
		x_pseudo = x_pseudo.view(ep_per_batch, n_way, n_pseudo, *img_shape)
		x_query = x_query.view(ep_per_batch, n_query, *img_shape)

		return x_shot, x_pseudo, x_query
