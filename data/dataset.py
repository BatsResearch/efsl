from PIL import Image
import numpy as np
import os
import pickle as pkl
from torch.utils.data import Dataset
from torchvision import transforms
import h5py as hp
import utils
import cv2 as cv


def trim_target_cls(src_classes, data_root, dataset, ilsvrc_int_id2unique_int_id, wnid2unique_int_id):
    if dataset in ['cifarfs', 'fc100']:
        trimmed = {int(cls): set() for cls in src_classes}
        with open(os.path.join(data_root, 'cifar_wnids.pkl'), 'rb') as f_:
            cifar_wnids = pkl.load(f_)
        for cls in src_classes:
            ids = list()
            for i in cifar_wnids[int(cls)]:
                if i in wnid2unique_int_id:
                    ids.append(wnid2unique_int_id.index(i))
            trimmed[int(cls)] = set(ids)
    else:
        trimmed = {int(cls): {ilsvrc_int_id2unique_int_id[int(cls)]} for cls in src_classes}

    return trimmed


def trim_cls(name, split, aux_classes, wnid2unique_int_id, height, dataset_dir):
    trimmed = dict()
    aux_cls = set([wnid2unique_int_id[item] for item in list(aux_classes)])

    statfolder = dataset_dir
    with open(os.path.join(statfolder, f'{name}-pruning.pkl'), 'rb') as f:
        pruning = pkl.load(f)
    pruning = pruning[split][f"l{height}"]
    for key, value in pruning.items():
        curr_trimmed = value['trimmed']
        curr_trimmed = iter(aux_cls.intersection(set(curr_trimmed)))
        trimmed[key] = [wnid2unique_int_id.index(item) for item in curr_trimmed]
    return trimmed

def get_imagenet_dir_tree(imagenet_folder_path, tree_path):
    if os.path.exists(tree_path):
        imagenet_dir_tree = pkl.load(open(tree_path, 'rb'))
        return imagenet_dir_tree
    else:
        all_classes = os.listdir(imagenet_folder_path)
        imagenet_dir_tree = dict()
        for cls_ in all_classes:
            imagenet_dir_tree[cls_] = os.listdir(os.path.join(imagenet_folder_path, cls_))

        with open(tree_path, 'wb') as f:
            pkl.dump(imagenet_dir_tree, f)

        return imagenet_dir_tree


class CustomDataset(Dataset):

    def __init__(self, name, split, data_dir, min_class_members=500, aux_data='imagenet', top_k=1, **kwargs):
        np.random.seed(0)
        augment = kwargs.get('augment')
        self.top_k = top_k
        self.aux_level = kwargs['aux_level']
        print(f"top k classes: {top_k}")
        dataset_dir = os.path.realpath(os.path.join(data_dir, name))

        self.imagenet_path = os.path.realpath(os.path.join(data_dir, 'imagenet_folder'))

        self.ilsvrc_int_id2unique_int_id = np.asarray(pkl.load(open(os.path.join(data_dir, 'ilsvrc_int_id2unique_int_id.pkl'), 'rb')))
        self.imagenet_int_id2unique_int_id = pkl.load(open(os.path.join(data_dir, 'imagenet_int_id2unique_int_id.pkl'), 'rb'))
        self.wnid2unique_int_id = pkl.load(open(os.path.join(data_dir, 'wnid2unique_int_id.pkl'), 'rb'))
        imagenet_dir_tree = get_imagenet_dir_tree(imagenet_folder_path=self.imagenet_path,
                                                  tree_path=os.path.realpath(os.path.join(data_dir, 'imagenet_dir_tree.pkl')))
        self.imagenet_dir_tree = dict()
        for key, value in imagenet_dir_tree.items():
            kk = self.imagenet_int_id2unique_int_id.index(self.wnid2unique_int_id.index(key))
            self.imagenet_dir_tree[str(kk)] = [(key, v) for v in value]

        aux_keys = list()
        for key, value in self.imagenet_dir_tree.items():
            if len(value) >= min_class_members:
                aux_keys.append(int(key))

        self.imagenet_int_id2unique_int_id = np.asarray(self.imagenet_int_id2unique_int_id)
        self.aux_to_unique_id = lambda x: self.imagenet_int_id2unique_int_id[x]

        dataset_fp = hp.File(os.path.realpath(os.path.join(dataset_dir, f"{name}-{split}.h5")), 'r')
        self.similarity_matrix = np.load(os.path.join(dataset_dir, f'conceptnet_similarities.npy'))

        dataset_keys = list(set(list(map(int, dataset_fp.keys()))))
        self.allowed_aux_classes = set(self.aux_to_unique_id(aux_keys))

        if split not in ['test', 'val']:
            trimmed_classes = list()

            test_trimmed = trim_cls(name=name, split='test', aux_classes=self.allowed_aux_classes, wnid2unique_int_id=self.wnid2unique_int_id, height=self.aux_level, dataset_dir=dataset_dir)
            trimmed_classes.append(test_trimmed)
            val_trimmed = trim_cls(name=name, split='val', aux_classes=self.allowed_aux_classes, wnid2unique_int_id=self.wnid2unique_int_id, height=self.aux_level, dataset_dir=dataset_dir)
            trimmed_classes.append(val_trimmed)
            self.forbidden_cls = trim_target_cls(dataset_keys, data_dir, name, self.ilsvrc_int_id2unique_int_id, self.wnid2unique_int_id)

            all_fb = set()
            for trimmed_set in trimmed_classes:
                for key, value in trimmed_set.items():
                    all_fb = all_fb.union(set(value))
            self.allowed_aux_classes = self.allowed_aux_classes.difference(all_fb)

        else:
            test_trimmed = trim_cls(name=name, split=split, aux_classes=self.allowed_aux_classes, wnid2unique_int_id=self.wnid2unique_int_id, height=self.aux_level, dataset_dir=dataset_dir)
            self.forbidden_cls = test_trimmed

        self.allowed_aux_classes = list(self.allowed_aux_classes)
        self.dataset_classes = dataset_keys
        self.n_classes = len(self.dataset_classes)

        self.dataset = dict()
        self.dataset_class_members = dict()
        self.dataset_path = list()
        self.dataset_labels = list()


        class_counter = 0
        for i, (key, value) in enumerate(dataset_fp.items()):
            num_samples = value.shape[0]
            int_key = int(key)
            self.dataset_class_members[int_key] = np.arange(num_samples) + len(self.dataset_path)
            self.dataset_path.extend([(int_key, k) for k in range(num_samples)])
            self.dataset_labels.extend([int_key] * num_samples)
            self.dataset[int_key] = value
            class_counter += 1

        self.allowed_aux_classes = set(self.allowed_aux_classes)

        self.aux_dataset = dict()
        self.aux_dataset_class_members = dict()
        self.aux_dataset_path = list()
        self.aux_dataset_labels = list()

        for key, value in self.imagenet_dir_tree.items():
            num_samples = len(value)
            int_key = int(key)
            unique_key = self.aux_to_unique_id(int_key)
            self.aux_dataset_class_members[unique_key] = np.arange(num_samples) + len(self.aux_dataset_path)
            self.aux_dataset_path.extend(value)
            self.aux_dataset_labels.extend([int(key)] * num_samples)
            class_counter += 1

        imagenet_norm_params = {'mean': [120.39586422 / 255.0, 115.59361427 / 255.0, 104.54012653 / 255.0],
                                'std': [70.68188272 / 255.0, 68.27635443 / 255.0, 72.54505529 / 255.0]}
        imagenet_normalize = transforms.Normalize(**imagenet_norm_params)

        src_normalize = imagenet_normalize
        aux_normalize = imagenet_normalize
        src_init_t = aux_init_t = transforms.Compose([lambda x: Image.fromarray(x)])

        if name in ['mini-imagenet', 'tiered-imagenet']:
            image_size = 84
            padding = 8
        elif name == 'im800':
            image_size = 256
            padding = 16
        elif name in ['cifarfs', 'fc100']:
            image_size = 32
            padding = 4
            cifar_norm_params = {'mean': [0.5071, 0.4867, 0.4408],
                                 'std': [0.2675, 0.2565, 0.2761]}
            cifar_normalize = transforms.Normalize(**cifar_norm_params)
            src_normalize = cifar_normalize
            aux_init_t = transforms.Compose([lambda x: Image.fromarray(x), transforms.Resize(image_size)])

        else:
            image_size = padding = None
            print('There is no such dataset.')
            exit()

        if augment:
            template_transform = transforms.Compose([
                transforms.RandomCrop(image_size, padding=padding),
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ])
            self.src_transform = transforms.Compose([src_init_t, template_transform, src_normalize])
            self.aux_transform = transforms.Compose([aux_init_t, template_transform, aux_normalize])
        else:
            self.src_transform = transforms.Compose([src_init_t, transforms.ToTensor(), src_normalize])
            self.aux_transform = transforms.Compose([aux_init_t, transforms.ToTensor(), aux_normalize])

        if split == 'test' and kwargs.get('orig_img'):
            self.label_func = self.test_label
        else:
            self.label_func = self.non_test_label

        self.src_transform_func = self.src_transform_func_no_aux

    def __len__(self):
        return 1

    def test_label(self, img, label):
        return img


    def non_test_label(self, img, label):
        return label

    def src_transform_func_no_aux(self, img, index):
        img = self.src_transform(img)
        return img

    def get_image(self, index_):

        if index_ < 0:
            index = -index_
            tp = self.aux_dataset_path[index]
            image_file_path = os.path.join(self.imagenet_path, tp[0], tp[1])
            image_file = cv.imread(image_file_path)
            image_file = cv.resize(image_file, (84, 84), interpolation=cv.INTER_CUBIC)
            raw_img = np.flip(image_file, -1)
            img = self.aux_transform(raw_img)
            label = self.aux_dataset_labels[index]
        else:
            tp = self.dataset_path[index_]
            raw_img = self.dataset[tp[0]][tp[1]][:]
            img = self.src_transform_func(raw_img, index_)
            label = self.dataset_labels[index_]
        return img, self.label_func(raw_img, label)

    def __getitem__(self, index_):
        image, embedding = self.get_image(index_)
        return image, embedding





class TrainDataset(Dataset):

    def __init__(self, name, split, data_dir, min_class_members=500, aux_data='imagenet', top_k=1, **kwargs):
        np.random.seed(0)
        augment = kwargs.get('augment')
        dataset_dir = os.path.realpath(os.path.join(data_dir, name))

        self.imagenet_path = os.path.realpath(os.path.join(data_dir, 'imagenet_folder'))

        dataset_fp = hp.File(os.path.realpath(os.path.join(dataset_dir, f"{name}-{split}.h5")), 'r')
        dataset_keys = list(set(list(map(int, dataset_fp.keys()))))
        self.ilsvrc_int_id2unique_int_id = np.asarray(pkl.load(open(os.path.join(data_dir, 'ilsvrc_int_id2unique_int_id.pkl'), 'rb')))
        self.imagenet_int_id2unique_int_id = pkl.load(open(os.path.join(data_dir, 'imagenet_int_id2unique_int_id.pkl'), 'rb'))
        self.wnid2unique_int_id = pkl.load(open(os.path.join(data_dir, 'wnid2unique_int_id.pkl'), 'rb'))
        imagenet_dir_tree = get_imagenet_dir_tree(imagenet_folder_path=self.imagenet_path,
                                                  tree_path=os.path.realpath(os.path.join(data_dir, 'imagenet_dir_tree.pkl')))
        self.imagenet_dir_tree = dict()
        for key, value in imagenet_dir_tree.items():
            kk = self.imagenet_int_id2unique_int_id.index(self.wnid2unique_int_id.index(key))
            self.imagenet_dir_tree[str(kk)] = [(key, v) for v in value]

        aux_keys = list()
        for key, value in self.imagenet_dir_tree.items():
            if len(value) >= min_class_members:
                aux_keys.append(int(key))

        self.imagenet_int_id2unique_int_id = np.asarray(self.imagenet_int_id2unique_int_id)
        self.aux_to_unique_id = lambda x: self.imagenet_int_id2unique_int_id[x]


        self.classes_in_experiment = {'base': dataset_keys, 'aux': list()}
        self.allowed_aux_classes = set(self.aux_to_unique_id(aux_keys))
        self.similarity_matrix = np.load(os.path.join(dataset_dir, f'conceptnet_similarities.npy'))

        self.aux_level = kwargs.get('aux_level')

        trimmed_classes = list()
        if split not in ['test', 'val']:
            test_height = 0 if kwargs.get('aux_level') is None else kwargs['aux_level']

            test_trimmed = trim_cls(name=name, split='test', aux_classes=self.allowed_aux_classes, wnid2unique_int_id=self.wnid2unique_int_id, height=test_height, dataset_dir=dataset_dir)
            trimmed_classes.append(test_trimmed)
            val_trimmed = trim_cls(name=name, split='val', aux_classes=self.allowed_aux_classes, wnid2unique_int_id=self.wnid2unique_int_id, height=test_height, dataset_dir=dataset_dir)
            trimmed_classes.append(val_trimmed)

            train_trimmed = trim_target_cls(dataset_keys, data_dir, name, self.ilsvrc_int_id2unique_int_id, self.wnid2unique_int_id)
            trimmed_classes.append(train_trimmed)
        else:
            test_height = 0 if kwargs.get('aux_level') is None else kwargs['aux_level']
            test_trimmed = trim_cls(name=name, split=split, aux_classes=self.allowed_aux_classes, wnid2unique_int_id=self.wnid2unique_int_id, height=test_height, dataset_dir=dataset_dir)
            trimmed_classes.append(test_trimmed)



        all_fb = set()
        for trimmed_set in trimmed_classes:
            for key, value in trimmed_set.items():
                all_fb = all_fb.union(set(value))

        utils.log(f"num allowed before trimming: {len(self.allowed_aux_classes)}")
        self.allowed_aux_classes = self.allowed_aux_classes.difference(all_fb)
        utils.log(f"num allowed after trimming: {len(self.allowed_aux_classes)}")
        self.allowed_aux_classes = list(self.allowed_aux_classes)
        self.dataset = dict()
        self.dataset_class_members = dict()
        self.dataset_path = list()
        self.dataset_labels = list()
        class_counter = 0


        for key, value in dataset_fp.items():
            num_samples = value.shape[0]
            self.dataset_path.extend([(class_counter, k) for k in range(num_samples)])
            self.dataset_labels.extend([class_counter] * num_samples)
            self.dataset[class_counter] = value
            class_counter += 1

        self.src_image_count = len(self.dataset_path)

        rows = np.asarray(dataset_keys)
        cols = np.asarray(self.allowed_aux_classes)
        similarity_matrix = (self.similarity_matrix.ravel()[(
                cols + (rows * self.similarity_matrix.shape[1]).reshape((-1, 1))
        ).ravel()]).reshape(rows.size, cols.size)

        chosen_indices = np.argpartition(similarity_matrix, axis=1, kth=-min(similarity_matrix.shape[1], (top_k)))[:, -min(similarity_matrix.shape[1], (top_k)):]
        if top_k == 0 or (kwargs.get('no_train_ps') is True):
            chosen_indices = list()

        chosen_cls = list(set(np.asarray(self.allowed_aux_classes)[chosen_indices].flatten()))
        self.classes_in_experiment['aux'] = chosen_cls

        for key, value in self.imagenet_dir_tree.items():
            int_key = int(key)
            unique_key = self.aux_to_unique_id(int_key)
            if unique_key not in chosen_cls:
                continue
            num_samples = len(value)
            self.dataset_path.extend(value)
            self.dataset_labels.extend([class_counter] * num_samples)
            class_counter += 1

        self.n_classes = class_counter
        imagenet_norm_params = {'mean': [120.39586422 / 255.0, 115.59361427 / 255.0, 104.54012653 / 255.0],
                                'std': [70.68188272 / 255.0, 68.27635443 / 255.0, 72.54505529 / 255.0]}
        imagenet_normalize = transforms.Normalize(**imagenet_norm_params)

        src_normalize = imagenet_normalize
        aux_normalize = imagenet_normalize
        src_init_t = aux_init_t = transforms.Compose([lambda x: Image.fromarray(x)])

        if name in ['mini-imagenet', 'tiered-imagenet']:
            image_size = 84
            padding = 8
        elif name == 'im800':
            image_size = 256
            padding = 16
        elif name in ['cifarfs', 'fc100']:
            image_size = 32
            padding = 4
            cifar_norm_params = {'mean': [0.5071, 0.4867, 0.4408],
                                 'std': [0.2675, 0.2565, 0.2761]}
            cifar_normalize = transforms.Normalize(**cifar_norm_params)
            src_normalize = cifar_normalize
            aux_init_t = transforms.Compose([lambda x: Image.fromarray(x), transforms.Resize(image_size)])

        else:
            image_size = padding = None
            print('There is no such dataset.')
            exit()

        if augment:
            template_transform = transforms.Compose([
                transforms.RandomCrop(image_size, padding=padding),
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ])
            self.src_transform = transforms.Compose([src_init_t, template_transform, src_normalize])
            self.aux_transform = transforms.Compose([aux_init_t, template_transform, aux_normalize])
        else:
            self.src_transform = transforms.Compose([src_init_t, transforms.ToTensor(), src_normalize])
            self.aux_transform = transforms.Compose([aux_init_t, transforms.ToTensor(), aux_normalize])

    def __len__(self):
        return len(self.dataset_labels)


    def get_image(self, index_):

        tp = self.dataset_path[index_]
        if index_ < self.src_image_count:
            img = self.dataset[tp[0]][tp[1]][:]
            img = self.src_transform(img)
        else:
            image_file_path = os.path.join(self.imagenet_path, tp[0], tp[1])
            image_file = cv.imread(image_file_path)
            image_file = cv.resize(image_file, (84, 84), interpolation=cv.INTER_CUBIC)
            raw_img = np.flip(image_file, -1)
            img = self.aux_transform(raw_img)

        label = self.dataset_labels[index_]
        return img, label

    def __getitem__(self, index_):
        image, label = self.get_image(index_)
        return image, label

