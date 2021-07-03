import utils
from data.dataset import CustomDataset
from data.sampler import EpisodicSampler
from data.dataset import TrainDataset
from torch.utils.data import DataLoader


def get_fs_loader(sub_dsname, config):
    if config.get(sub_dsname) is not None:
        dataset = CustomDataset(config[sub_dsname], **config[f'{sub_dsname}_args'], data_dir=config['data_dir'])
        utils.log(f"{sub_dsname.replace('_', ' ')}-{config['n_shot']}shot, shape: {dataset[0][0].shape}, aux-level: {dataset.aux_level}")

        sampler = EpisodicSampler(
            dataset=dataset,
            n_batch=config['num_batches'],
            n_way=config['n_way'],
            n_shot=config['n_shot'],
            n_query=config['n_query'],
            n_pseudo=config['n_pseudo'],
            episodes_per_batch=config['ep_per_batch']
        )

        dataloader = DataLoader(dataset, batch_sampler=sampler, num_workers=config['workers'],
                                pin_memory=True)

    else:
        dataloader = None

    return dataloader


def get_classification_loader(sub_dsname, config):
    dataset = TrainDataset(name=config[sub_dsname], **config[f'{sub_dsname}_args'], data_dir=config['data_dir'])
    loader = DataLoader(dataset, config['batch_size'], shuffle=True,
                        num_workers=config['workers'], pin_memory=True, drop_last=True)

    utils.log(f"{sub_dsname.replace('_', ' ')}, shape: {dataset[0][0].shape}, classes: {dataset.n_classes}, aux-level: {dataset.aux_level}")

    return loader, dataset.n_classes
