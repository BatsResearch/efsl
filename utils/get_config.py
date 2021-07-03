import argparse
import os
import yaml
import uuid
import sys
import datetime
import utils


def get_config_train_fsl():
    parser = argparse.ArgumentParser()

    # Model Arguments
    parser.add_argument('--model', type=str, default='cosine', choices=['cosine', 'relation-net', 'feat'])
    parser.add_argument('--model-checkpoint', type=str, default=None, help='path to saved model params')
    parser.add_argument('--encoder-config', required=True, type=str, help='path to encoder config (resnet12.yaml/resnet12-mask.yaml')
    parser.add_argument('--encoder-checkpoint', default=None, type=str, help='path to saved encoder params')
    parser.add_argument('--inner-encoder', action='store_true', help='set true when encoder is resnet12-mask to only load the params of resnet')


    # Data Arguments
    parser.add_argument('--dataset', type=str, default='mini-imagenet', choices=['mini-imagenet', 'tiered-imagenet', 'cifarfs', 'fc100'])
    parser.add_argument('--topk', type=int, default=3, help='choose from top K most similar classes')
    parser.add_argument('--aux-level', type=int, default=0, choices=[0, 1, 2, 3], help='level of auxiliary data (pruning)')
    parser.add_argument('--data-dir', type=str, default='./../data_root_h5', help='path to datasets')

    # Data loader args
    parser.add_argument('--nway', type=int, default=5)
    parser.add_argument('--nshot', type=int, default=1)
    parser.add_argument('--npseudo', type=int, default=15, help='number of pseudo shots for each class')
    parser.add_argument('--nquery', type=int, default=15)
    parser.add_argument('--num-batches', type=int, default=200, help='Number of batches in each epoch.')
    parser.add_argument('--batch-size', type=int, default=4, help='Number of episodes in each batch.')

    # optimization args

    parser.add_argument('--optimizer', type=str, default='sgd', choices=['sgd', 'adam'])
    parser.add_argument('--epochs', type=int, default=150, help='max training epochs')
    parser.add_argument('--lr', type=float, default=0.05)
    parser.add_argument('--decay', type=float, default=5.e-4)
    parser.add_argument('--milestones', nargs='+', type=int, default=[70])


    # Others
    parser.add_argument('--save-epoch', type=int, default=5, help='Frequency of checkpoints')
    parser.add_argument('--save-dir', type=str, default='save', help='directory for saving checkpoints')
    parser.add_argument('--train-encoder', action='store_true', help='set true to unfreeze the parameters of the encoder')
    parser.add_argument('--freeze-bn', action='store_true')
    parser.add_argument('--name', default=None, type=str, help='optional name for experiment')
    parser.add_argument('--workers', type=int, default=4, help='num parallel data loaders')
    parser.add_argument('--tag', type=str, default=None, help='extension to the experiment name')


    args = parser.parse_args()
    config = dict()

    config['model'] = args.model
    config['model_args'] = yaml.load(open(args.encoder_config, 'r'), Loader=yaml.FullLoader)
    config['encoder_checkpoint'] = args.encoder_checkpoint
    config['model_checkpoint'] = args.model_checkpoint
    config['inner_encoder'] = args.inner_encoder

    config['dataset'] = args.dataset
    config['data_dir'] = args.data_dir
    for split in ['train', 'test', 'val']:
        config[f'{split}_dataset'] = args.dataset
        config[f'{split}_dataset_args'] = {'split': split, 'augment': bool(split == 'train'), 'top_k': args.topk, 'aux_level': args.aux_level}

    config['n_way'] = args.nway
    config['n_shot'] = args.nshot
    config['n_pseudo'] = args.npseudo
    config['n_query'] = args.nquery
    config['num_batches'] = args.num_batches
    config['ep_per_batch'] = args.batch_size


    config['optimizer'] = args.optimizer
    config['optimizer_args'] = {'lr': args.lr, 'weight_decay': args.decay, 'milestones': args.milestones}
    config['max_epoch'] = args.epochs

    config['freeze_encoder'] = not args.train_encoder
    config['freeze_bn'] = args.freeze_bn
    config['save_epoch'] = args.save_epoch
    config['workers'] = args.workers

    if args.name is not None:
        config['name'] = args.name
    else:
        random_id = uuid.uuid4().hex[:15].lower()
        config['uuid'] = random_id
        tag = f"-{args.tag}" if args.tag is not None else ""
        name = f"{config['model']}-{os.path.basename(args.encoder_config).replace('.yaml', '')}-{args.dataset}-{args.nshot}shot-aux{args.aux_level}-ps{args.npseudo}{tag}-{random_id}"
        config['name'] = name


    if args.dataset in ['cifarfs', 'fc100']:
        config = utils.update_dictionary_entry(config, 'dropblock_size', 2)


    command = sys.argv
    command[0] = os.path.basename(command[0])
    command = ' '.join(command)

    return config, command, args.save_dir



def get_config_test():
    parser = argparse.ArgumentParser()

    # Model Arguments
    parser.add_argument('--model', type=str, default='cosine', choices=['cosine', 'relation-net', 'feat'])
    parser.add_argument('--model-checkpoint', type=str, default=None, help='path to saved model params')
    parser.add_argument('--encoder-config', required=True, type=str, help='path to encoder config (resnet12.yaml/resnet12-mask.yaml')
    parser.add_argument('--encoder-checkpoint', default=None, type=str, help='path to saved encoder params')
    parser.add_argument('--inner-encoder', action='store_true', help='set true when encoder is resnet12-mask to only load the params of resnet')


    # Data Arguments
    parser.add_argument('--dataset', type=str, default='mini-imagenet', choices=['mini-imagenet', 'tiered-imagenet', 'cifarfs', 'fc100'])
    parser.add_argument('--topk', type=int, default=3, help='choose from top K most similar classes')
    parser.add_argument('--aux-level', type=int, default=0, choices=[0, 1, 2, 3], help='level of auxiliary data (pruning)')
    parser.add_argument('--data-dir', type=str, default='./../data_root_h5', help='path to datasets')

    # Data loader args
    parser.add_argument('--nway', type=int, default=5)
    parser.add_argument('--nshot', type=int, default=1)
    parser.add_argument('--npseudo', type=int, default=15, help='number of pseudo shots for each class')
    parser.add_argument('--nquery', type=int, default=15)
    parser.add_argument('--num-batches', type=int, default=200, help='Number of batches in each epoch.')
    parser.add_argument('--batch-size', type=int, default=4, help='Number of episodes in each batch.')


    # Others
    parser.add_argument('--train-encoder', action='store_true')
    parser.add_argument('--random', action='store_true')
    parser.add_argument('--raw-log', action='store_true')
    parser.add_argument('--log-freq', type=int, default=100)
    parser.add_argument('--log-dir', type=str, default='log_test')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--tag', type=str, default=None)
    parser.add_argument('--name', type=str, default=None)


    args = parser.parse_args()
    config = dict()

    config['model'] = args.model
    config['model_args'] = yaml.load(open(args.encoder_config, 'r'), Loader=yaml.FullLoader)
    config['encoder_checkpoint'] = args.encoder_checkpoint
    config['model_checkpoint'] = args.model_checkpoint
    config['inner_encoder'] = args.inner_encoder

    config['dataset'] = args.dataset
    config['data_dir'] = args.data_dir
    for split in ['test']:
        config[f'{split}_dataset'] = args.dataset
        if args.random:
            config[f'{split}_dataset_args'] = {'split': split, 'augment': False, 'top_k': args.topk, 'aux_level': args.aux_level, 'random': True}
        else:
            config[f'{split}_dataset_args'] = {'split': split, 'augment': False, 'top_k': args.topk, 'aux_level': args.aux_level}

    config['n_way'] = args.nway
    config['n_shot'] = args.nshot
    config['n_pseudo'] = args.npseudo
    config['n_query'] = args.nquery
    config['num_batches'] = args.num_batches
    config['ep_per_batch'] = args.batch_size

    config['freeze_encoder'] = not args.train_encoder
    config['workers'] = args.workers
    config['log_freq'] = args.log_freq
    config['raw_log'] = args.raw_log



    if args.name is not None:
        config['name'] = args.name
    else:
        dir_with_uuid = 'model_checkpoint' if config['model_checkpoint'] is not None else 'encoder_checkpoint'
        random_id = os.path.dirname(dir_with_uuid).split('-')[-1]
        tag = f"-{args.tag}" if args.tag is not None else ""
        name = f"{config['model']}-{os.path.basename(args.encoder_config).replace('.yaml', '')}-{args.dataset}-{args.nshot}shot-aux{args.aux_level}-ps{args.npseudo}{tag}-{random_id}"
        config['name'] = name

    now = datetime.datetime.now()
    timestamp = now.strftime('%Y-%m-%dT%H-%M-%S') + ('-%02d' % (now.microsecond / 10000))
    config['log_dir'] = os.path.join(args.log_dir, config['name'] + f"-{timestamp}")

    command = sys.argv
    command[0] = os.path.basename(command[0])
    command = ' '.join(command)

    if args.dataset in ['cifarfs', 'fc100']:
        config = utils.update_dictionary_entry(config, 'dropblock_size', 2)

    return config, command



def get_config_classifier():
    parser = argparse.ArgumentParser()

    # Model Arguments
    parser.add_argument('--config', type=str, required=True, help='path classifier config file')
    parser.add_argument('--model-checkpoint', type=str, default=None, help='path to saved model params')


    # Data Arguments
    parser.add_argument('--dataset', type=str, default='mini-imagenet', choices=['mini-imagenet', 'tiered-imagenet', 'cifarfs', 'fc100'])
    parser.add_argument('--topk', type=int, default=3, help='choose from top K most similar classes')
    parser.add_argument('--aux-level', type=int, default=0, choices=[0, 1, 2, 3], help='level of auxiliary data (pruning)')
    parser.add_argument('--data-dir', type=str, default='./../data_root_h5')
    parser.add_argument('--no-train-ps', action='store_true', help='path to datasets')
    # optimization args
    parser.add_argument('--batch-size', type=int, default=None)
    parser.add_argument('--epochs', type=int, default=120)
    parser.add_argument('--optimizer', type=str, default='sgd', choices=['sgd', 'adam'])
    parser.add_argument('--lr', type=float, default=0.05)
    parser.add_argument('--decay', type=float, default=5.e-4)
    parser.add_argument('--milestones', nargs='+', type=int, default=[60, 90])

    parser.add_argument('--nway', type=int, default=5)
    parser.add_argument('--npseudo', type=int, default=15, help='number of pseudo shots for each class')
    parser.add_argument('--nquery', type=int, default=15)
    parser.add_argument('--num-fs-batches', type=int, default=200)
    parser.add_argument('--fs-batch-size', type=int, default=4)


    # Others
    parser.add_argument('--eval-classifier', action='store_true')
    parser.add_argument('--evalfs-epoch', type=int, default=5)
    parser.add_argument('--save-epoch', type=int, default=5)
    parser.add_argument('--save-dir', type=str, default='save', help='directory for saving checkpoints')
    parser.add_argument('--name', default=None, type=str)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--tag', type=str, default=None)


    args = parser.parse_args()
    config = dict()

    model_config = yaml.load(open(args.config, 'r'), Loader=yaml.FullLoader)
    config['model'] = model_config['model']
    config['model_args'] = model_config['model_args']

    config['model_checkpoint'] = args.model_checkpoint
    config['data_dir'] = args.data_dir

    config['train_dataset'] = args.dataset
    config['train_dataset_args'] = {'split': 'train', 'augment': True, 'top_k': args.topk, 'aux_level': args.aux_level, 'no_train_ps': args.no_train_ps}
    if args.eval_classifier:
        config['val_dataset'] = args.dataset
        config['val_dataset_args'] = {'split': 'val', 'augment': False, 'top_k': args.topk, 'aux_level': args.aux_level,
                                        'no_train_ps': args.no_train_ps}

    config['no_train_ps'] = args.no_train_ps

    config['n_way'] = args.nway
    config['n_pseudo'] = args.npseudo
    config['n_query'] = args.nquery
    config['num_batches'] = args.num_fs_batches
    config['ep_per_batch'] = args.fs_batch_size

    config['dataset'] = args.dataset
    for split in ['test', 'val']:
        config[f'fs_{split}_dataset'] = args.dataset
        config[f'fs_{split}_dataset_args'] = {'split': split, 'augment': False, 'top_k': args.topk, 'aux_level': args.aux_level}

    config['eval_fs_epoch'] = args.evalfs_epoch


    config['optimizer'] = args.optimizer
    config['optimizer_args'] = {'lr': args.lr, 'weight_decay': args.decay, 'milestones': args.milestones}

    config['max_epoch'] = args.epochs

    if args.batch_size is not None:
        config['batch_size'] = args.batch_size
    else:
        config['batch_size'] = 256 if args.dataset == 'tiered-imagenet' else 64

    config['save_epoch'] = args.save_epoch
    config['workers'] = args.workers

    if args.name is not None:
        config['name'] = args.name
    else:
        random_id = uuid.uuid4().hex[:15].lower()
        config['uuid'] = random_id
        tag = f"-{args.tag}" if args.tag is not None else ""
        no_ps = f"-no_ps" if args.no_train_ps else ""
        name = f"{config['model']}-{args.dataset}{no_ps}-aux{args.aux_level}-{tag}-{random_id}"
        config['name'] = name


    command = sys.argv
    command[0] = os.path.basename(command[0])
    command = ' '.join(command)

    if args.dataset in ['cifarfs', 'fc100']:
        config = utils.update_dictionary_entry(config, 'dropblock_size', 2)

    return config, command, args.save_dir
