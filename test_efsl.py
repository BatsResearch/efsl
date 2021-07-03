import torch
import torch.nn as nn
from tqdm import tqdm
import os
import pickle as pkl

import utils
import utils.few_shot as fs

from utils.get_data_loader import get_fs_loader
from utils.get_model import get_model
from utils.get_config import get_config_test as get_config
from pathlib import Path


def main(config):

    if config['raw_log']:
        config['test_dataset_args']['orig_img'] = True
    loader = get_fs_loader('test_dataset', config)

    config['model_args']['input_shape'] = list(loader.dataset[0][0].shape)

    model = get_model(config)

    if torch.cuda.device_count() > 1:
        is_parallel = True
        model = nn.DataParallel(model)
    else:
        is_parallel = False


    model.eval()
    # testing
    aves_keys = ['vl', 'va']
    aves = {k: utils.Averager() for k in aves_keys}

    va_lst = []
    raw_data = []
    for epoch in range(1, 2):
        for i, (data, _) in tqdm(enumerate(loader), desc=f"eval: ", total=len(loader), leave=True):
            log_flag = (not (i % config['log_freq'])) and config['raw_log']

            with torch.no_grad():

                logits, acc, loss, raw_log = fs.predict(
                    model=model,
                    data=data,
                    n_way=config['n_way'],
                    n_shot=config['n_shot'],
                    n_query=config['n_query'],
                    n_pseudo=config['n_pseudo'],
                    ep_per_batch=config['ep_per_batch'],
                    return_log=log_flag
                )

            if log_flag:
                raw_data.append(raw_log)
            aves['vl'].add(loss.item(), len(data))
            aves['va'].add(acc, len(data))
            va_lst.append(acc)

        utils.log(f"Test, acc: {aves['va'].item() * 100: .2f} +- "
                  f"{utils.mean_confidence_interval(va_lst) * 100: .2f}, "
                  f"loss: {aves['vl'].item(): .2f}")



    config['val_based_best'] = {'acc': aves['va'].item() * 100, 'err': utils.mean_confidence_interval(va_lst) * 100,'loss': aves['vl'].item()}


    if config['raw_log']:
        with open(os.path.join(config['log_dir'], 'raw_data.pkl'), 'wb') as f:
            pkl.dump(raw_data, f)

if __name__ == '__main__':

    config, command = get_config()

    Path(config['log_dir']).mkdir(exist_ok=True, parents=True)

    utils.set_log_path(config['log_dir'])
    utils.set_log_name(f"test_log.txt")
    utils.log(command + '\n')

    main(config)