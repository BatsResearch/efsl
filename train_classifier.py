import os
import yaml

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from tensorboardX import SummaryWriter

import utils
import utils.few_shot as fs

from utils.get_data_loader import get_fs_loader, get_classification_loader
from utils.get_model import get_model
from utils.get_config import get_config_classifier as get_config


def main(config, command, save_dir):

    save_path = os.path.join(save_dir, config['name'])
    utils.ensure_path(save_path)
    utils.set_log_path(save_path)

    with open(os.path.join(save_path, 'command.txt'), 'w') as f:
        print(command, file=f)

    utils.log(config['name'])
    writer = SummaryWriter(os.path.join(save_path, 'tensorboard'))


    train_loader, n_classes = get_classification_loader(sub_dsname='train_dataset', config=config)
    config['model_args']['classifier_args']['n_classes'] = n_classes

    if config.get('val_dataset') is not None:
        val_loader, _ = get_classification_loader(sub_dsname='val_dataset', config=config)
    else:
        val_loader = None

    yaml.dump(config, open(os.path.join(save_path, 'config.yaml'), 'w'))


    # few-shot eval
    fs_loaders = dict()
    for s in ['test', 'val']:
        name_ = f"fs_{s}_dataset"
        if config.get(name_) is not None:
            fs_loaders[name_] = {nshot_: get_fs_loader(sub_dsname=name_, config={**config, 'n_shot': nshot_}) for nshot_ in [1, 5]}


    #### Model and Optimizer ####

    model = get_model(config)
    if len(fs_loaders) != 0:
        fs_model = get_model({'model': 'cosine', 'model_args': {'encoder': None}})
        fs_model.encoder = model.encoder
        fs_model.encoder_name = config['model_args']['encoder']

    if torch.cuda.device_count() > 1:
        is_parallel = True
        model = nn.DataParallel(model)
        if len(fs_loaders) != 0:
            fs_model = nn.DataParallel(fs_model)
    else:
        is_parallel = False

    utils.log('num params: {}'.format(utils.compute_n_params(model)))

    optimizer, lr_scheduler = utils.make_optimizer(
            model.parameters(),
            config['optimizer'], **config['optimizer_args'])

    ########
    
    max_va = 0.
    timer_used = utils.Timer()
    timer_epoch = utils.Timer()

    aves_keys = ['tl', 'ta', 'vl', 'va', 'fsat-1', 'fsat-5', 'fsav-1', 'fsav-5']

    for epoch in range(1, config['max_epoch'] + 1):
        timer_epoch.s()
        aves = {k: utils.Averager() for k in aves_keys}

        # train
        model.train()
        writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)

        for data, label in tqdm(train_loader, desc='train', leave=False):
            data, label = data.cuda(), label.cuda()
            logits = model(data)
            loss = F.cross_entropy(logits, label)
            acc = utils.compute_acc(logits, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            aves['tl'].add(loss.item())
            aves['ta'].add(acc)

            logits = None; loss = None

        # eval
        if val_loader is not None:
            model.eval()
            for data, label in tqdm(val_loader, desc='val', leave=False):
                data, label = data.cuda(), label.cuda()
                with torch.no_grad():
                    logits = model(data)
                    loss = F.cross_entropy(logits, label)
                    acc = utils.compute_acc(logits, label)
                
                aves['vl'].add(loss.item())
                aves['va'].add(acc)

        if not epoch % config['eval_fs_epoch']:
            for key, value_ in fs_loaders.items():
                s = key.split('_')[1]
                for nshot__, loader in value_.items():
                    np.random.seed(0)
                    for data, _ in tqdm(loader, desc=f"fs-{s}-{nshot__}", leave=False):
                        with torch.no_grad():

                            logits, acc, _, _ = fs.predict(
                                model=fs_model,
                                data=data,
                                n_way=config['n_way'],
                                n_shot=nshot__,
                                n_query=config['n_query'],
                                n_pseudo=config['n_pseudo'],
                                ep_per_batch=config['ep_per_batch'],
                            )


                        aves['fsa' + s[0] + '-' + str(nshot__)].add(acc)


        # post
        if lr_scheduler is not None:
            lr_scheduler.step()

        for k, v in aves.items():
            aves[k] = v.item()

        t_epoch = utils.time_str(timer_epoch.t())
        t_used = utils.time_str(timer_used.t())
        t_estimate = utils.time_str(timer_used.t() / epoch * config['max_epoch'])

        if epoch <= config['max_epoch']:
            epoch_str = str(epoch)
        else:
            epoch_str = 'ex'
        log_str = 'epoch {}, train {:.4f}|{:.4f}'.format(
                epoch_str, aves['tl'], aves['ta'])
        writer.add_scalars('loss', {'train': aves['tl']}, epoch)
        writer.add_scalars('acc', {'train': aves['ta']}, epoch)

        if val_loader is not None:
            log_str += ', val {:.4f}|{:.4f}'.format(aves['vl'], aves['va'])
            writer.add_scalars('loss', {'val': aves['vl']}, epoch)
            writer.add_scalars('acc', {'val': aves['va']}, epoch)

        if not epoch % config['eval_fs_epoch']:
            for key, value_ in fs_loaders.items():
                s = key.split('_')[1]
                tag = s[0]
                log_str += f" {s}: "
                for nshot in value_.keys():
                    key = 'fsa' + tag + '-' + str(nshot)
                    log_str += ' {}: {:.4f}'.format(nshot, aves[key])
                    writer.add_scalars('acc', {key: aves[key]}, epoch)


        if epoch <= config['max_epoch']:
            log_str += ', {} {}/{}'.format(t_epoch, t_used, t_estimate)
        else:
            log_str += ', {}'.format(t_epoch)
        utils.log(log_str)

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
        if epoch <= config['max_epoch']:
            torch.save(save_obj, os.path.join(save_path, 'epoch-last.pth'))

            if (config['save_epoch'] is not None) and epoch % config['save_epoch'] == 0:
                torch.save(save_obj, os.path.join(
                    save_path, 'epoch-{}.pth'.format(epoch)))

            if aves['va'] > max_va:
                max_va = aves['va']
                torch.save(save_obj, os.path.join(save_path, 'max-va.pth'))
        else:
            torch.save(save_obj, os.path.join(save_path, 'epoch-ex.pth'))

        writer.flush()


if __name__ == '__main__':

    config, command, save_dir = get_config()
    main(config, command, save_dir) #

