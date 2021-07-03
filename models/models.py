import torch


models = {}


def register(name):
    def decorator(cls):
        models[name] = cls
        return cls
    return decorator


def make(name, **kwargs):
    if name is None:
        return None
    model = models[name](**kwargs)
    if torch.cuda.is_available():
        model.cuda()
    return model


def load(model_sv, name=None):#, test_model=None):
    if name is None:
        name = 'model'
    # if test_model is not None:
    #     model = make(test_model, **model_sv[name + '_args'])
    # else:

    for key in ['train', 'test', 'val']:
        if model_sv['config'].get(f"{key}_dataset") in ['cifarfs', 'fc100'] and 'resnet' in model_sv['model_args']['encoder']:
            model_sv[name + '_args']['encoder_args']['dropblock_size'] = 2
            break

    model = make(model_sv[name], **model_sv[name + '_args'])

    # if not test_model:
    model.load_state_dict(model_sv[name + '_sd'])#, strict=False)
    # elif test_model is not None:
    #     model.load_state_dict(model_sv[name + '_sd'], strict=False)

    return model

