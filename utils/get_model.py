import torch
import torch.nn as nn
import models


def get_model(config):
    if config.get('model_checkpoint') is not None:
        saved_model = torch.load(config['model_checkpoint'])
        model = models.load(saved_model)
    else:
        model = models.make(name=config['model'], **config['model_args'])


        if config.get('encoder_checkpoint') is not None:
            if config['inner_encoder']:
                target_encoder = model.encoder.encoder
            else:
                target_encoder = model.encoder
            saved_file = torch.load(config['encoder_checkpoint'])
            saved_model = models.load(saved_file)
            saved_encoder = saved_model.encoder
            saved_encoder_state_dict = saved_encoder.state_dict()
            target_encoder.load_state_dict(saved_encoder_state_dict)

            if config['freeze_encoder']:
                for param in target_encoder.parameters():
                    param.requires_grad = False

    return model
