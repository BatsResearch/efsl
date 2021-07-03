import os
import yaml

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from tensorboardX import SummaryWriter

import utils
import utils.few_shot as fs

from utils.get_data_loader import get_fs_loader
from utils.get_model import get_model
from utils.get_config import get_config_train_fsl as get_config


def main(config, command, save_dir):
    save_path = os.path.join(save_dir, config['name'])
    utils.ensure_path(save_path)
    utils.set_log_path(save_path)

    with open(os.path.join(save_path, 'command.txt'), 'w') as f:
        print(command, file=f)

    utils.log(config['name'])
    writer = SummaryWriter(os.path.join(save_path, 'tensorboard'))

    yaml.dump(config, open(os.path.join(save_path, 'config.yaml'), 'w'))

    train_loader = get_fs_loader('train_dataset', config)
    test_loader = get_fs_loader('test_dataset', config)
    val_loader = get_fs_loader('val_dataset', config)

    config['model_args']['input_shape'] = list(train_loader.dataset[0][0].shape)


    #### Model and optimizer ####

    model = get_model(config)

    if torch.cuda.device_count() > 1:
        is_parallel = True
        model = nn.DataParallel(model)
    else:
        is_parallel = False


    utils.log('num params: {}'.format(utils.compute_n_params(model)))
    utils.log('num trainable params: {}'.format(utils.compute_n_trainable_params(model)))

    optimizer, lr_scheduler = utils.make_optimizer(
        model.parameters(),
        config['optimizer'], **config['optimizer_args'])

    ########

    max_va = 0.
    max_tva = 0.
    recent_maxes = {key: {k2: {k3: 0.0 for k3 in ['train', 'test', 'val']} for k2 in ['loss', 'acc', 'err', 'epoch']} for key in ['test_based_best', 'val_based_best']}
    for key in recent_maxes.keys():
        recent_maxes[key]['epoch'] = -1

    timer_used = utils.Timer()
    timer_epoch = utils.Timer()

    aves_keys = ['tl', 'ta', 'tvl', 'tva', 'vl', 'va']
    trlog = dict()
    for k in aves_keys:
        trlog[k] = []

    for epoch in range(1, config['max_epoch'] + 1):
        timer_epoch.s()
        aves = {k: utils.Averager() for k in aves_keys}
        acc_dump = {k: list() for k in aves_keys if k[-1] == 'a'}

        # train
        model.train()
        if config['freeze_bn']:
            utils.freeze_bn(model)

        writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)

        np.random.seed(epoch)

        for data, _ in tqdm(train_loader, desc='train', leave=False):

            logits, acc, loss, _ = fs.predict(
                model=model,
                data=data,
                n_way=config['n_way'],
                n_shot=config['n_shot'],
                n_query=config['n_query'],
                n_pseudo=config['n_pseudo'],
                ep_per_batch=config['ep_per_batch'],
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            aves['tl'].add(loss.item())
            aves['ta'].add(acc)
            acc_dump['ta'].append(acc)

            logits = None; loss = None

            # eval
        model.eval()
        for name, loader, name_l, name_a in [
            ('test', test_loader, 'tvl', 'tva'),
            ('val', val_loader, 'vl', 'va')]:

            if loader is None:
                continue

            np.random.seed(0)
            for data, _ in tqdm(loader, desc=name, leave=False):
                with torch.no_grad():
                    logits, acc, loss, _ = fs.predict(
                        model=model,
                        data=data,
                        n_way=config['n_way'],
                        n_shot=config['n_shot'],
                        n_query=config['n_query'],
                        n_pseudo=config['n_pseudo'],
                        ep_per_batch=config['ep_per_batch'],
                    )

                aves[name_l].add(loss.item())
                aves[name_a].add(acc)
                acc_dump[name_a].append(acc)

        # post
        if lr_scheduler is not None:
            lr_scheduler.step()

        for k, v in aves.items():
            aves[k] = v.item()
            trlog[k].append(aves[k])

        t_epoch = utils.time_str(timer_epoch.t())
        t_used = utils.time_str(timer_used.t())
        t_estimate = utils.time_str(timer_used.t() / epoch * config['max_epoch'])

        utils.log(f"Epoch: {epoch}, "
                  f"train {aves['tl']: .4f}|{aves['ta']: .4f}, "
                  f"val {aves['vl']: .4f}|{aves['va']: .4f}, "
                  f"test {aves['tvl']: .4f}|{aves['tva']: .4f}, "
                  f"best {recent_maxes['val_based_best']['acc']['test']: .2f} @ {recent_maxes['val_based_best']['epoch']}, "
                  f"{t_epoch} {t_used}/{t_estimate}")

        writer.add_scalars('loss', {
            'train': aves['tl'],
            'tval': aves['tvl'],
            'val': aves['vl'],
        }, epoch)
        writer.add_scalars('acc', {
            'train': aves['ta'],
            'tval': aves['tva'],
            'val': aves['va'],
        }, epoch)

        if is_parallel:
            model_ = model.module
        else:
            model_ = model

        training = {
            'epoch': epoch,
            'optimizer': config['optimizer'],
            'optimizer_args': config['optimizer_args'],
            'optimizer_sd': optimizer.state_dict(),
        }
        save_obj = {
            'file': __file__,
            'config': config,

            'model': config['model'],
            'model_args': config['model_args'],
            'model_sd': model_.state_dict(),

            'training': training,
        }

        torch.save(save_obj, os.path.join(save_path, 'epoch-last.pth'))
        torch.save(trlog, os.path.join(save_path, 'trlog.pth'))

        if (config['save_epoch'] != 0) and epoch % config['save_epoch'] == 0:
            torch.save(save_obj,
                       os.path.join(save_path, 'epoch-{}.pth'.format(epoch)))


        if aves['va'] > max_va:
            recent_maxes['val_based_best']['acc']['test'] = aves['tva'] * 100
            recent_maxes['val_based_best']['acc']['train'] = aves['ta'] * 100
            recent_maxes['val_based_best']['acc']['val'] = aves['va'] * 100
            recent_maxes['val_based_best']['loss']['test'] = aves['tvl']
            recent_maxes['val_based_best']['loss']['train'] = aves['tl']
            recent_maxes['val_based_best']['loss']['val'] = aves['vl']
            recent_maxes['val_based_best']['epoch'] = epoch
            max_va = aves['va']
            torch.save(save_obj, os.path.join(save_path, 'max-va.pth'))

        if aves['tva'] > max_tva:
            recent_maxes['test_based_best']['acc']['test'] = aves['tva'] * 100
            recent_maxes['test_based_best']['acc']['train'] = aves['ta'] * 100
            recent_maxes['test_based_best']['acc']['val'] = aves['va'] * 100
            recent_maxes['test_based_best']['loss']['test'] = aves['tvl']
            recent_maxes['test_based_best']['loss']['train'] = aves['tl']
            recent_maxes['test_based_best']['loss']['val'] = aves['vl']
            recent_maxes['test_based_best']['epoch'] = epoch
            max_tva = aves['tva']

        writer.flush()

    for key, value in recent_maxes.items():
        config[key] = value



if __name__ == '__main__':
    config, command, save_dir = get_config()

    main(config, command, save_dir)

