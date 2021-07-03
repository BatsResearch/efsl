import numpy as np
import torch


class EpisodicSampler:

    def __init__(self, dataset, n_batch, n_way, n_shot, n_query, n_pseudo, episodes_per_batch=1):
        self.dataset = dataset
        self.n_batch = n_batch
        self.n_way = n_way
        self.n_shot = n_shot
        self.n_query = n_query
        self.n_pseudo = n_pseudo
        self.episodes_per_batch = episodes_per_batch
        self.top_k = self.dataset.top_k

        self.class_members = self.dataset.dataset_class_members
        self.aux_class_members = self.dataset.aux_dataset_class_members
        self.all_class_ids = self.dataset.dataset_classes
        # self.dataset.similarity_matrix = self.dataset.similarity_matrix
        self.allowed_class_ids = self.dataset.allowed_aux_classes
        self.forbidden_cls = self.dataset.forbidden_cls

    def __len__(self):
        return self.n_batch

    def __iter__(self):
        for i_batch in range(self.n_batch):
            batch = []
            for i_ep in range(self.episodes_per_batch):
                episode = []
                classes = np.random.choice(self.all_class_ids, self.n_way, replace=False)
                allowed_classes = self.allowed_class_ids.difference(set().union(*[self.forbidden_cls[i] for i in classes]))
                allowed_classes = np.asarray(list(allowed_classes))

                rows = classes
                cols = allowed_classes
                sim_matrix = (self.dataset.similarity_matrix.ravel()[(
                        cols + (rows * self.dataset.similarity_matrix.shape[1]).reshape((-1, 1))
                ).ravel()]).reshape(rows.size, cols.size)

                # sim_matrix = self.dataset.similarity_matrix[:, allowed_classes]
                # sim_matrix = sim_matrix[classes, :]

                chosen_indices = np.argpartition(sim_matrix, axis=1, kth=-min(self.top_k, sim_matrix.shape[1]))[:, -min(self.top_k, sim_matrix.shape[1]):]
                # print("chosen indices", chosen_indices)

                # similar_indices = np.flip(np.argsort(sim_matrix, axis=1, kind='stable'), -1)
                # chosen_indices = similar_indices[:, :self.top_k]
                chosen_cls = np.asarray(allowed_classes)[chosen_indices]
                for c, p in zip(classes, chosen_cls):
                    l1 = np.random.choice(self.class_members[c], self.n_shot + self.n_query, replace=False)
                    pseudo_class_members = list()
                    if self.top_k != 1:
                        for t in p:
                            pseudo_class_members.extend(self.aux_class_members[t])
                    else:
                        pseudo_class_members = self.aux_class_members[p[0]]
                    l2 = -np.random.choice(pseudo_class_members, self.n_pseudo, replace=False)
                    all_ = np.concatenate((l1, l2), axis=0)
                    episode.append(torch.from_numpy(all_))
                episode = torch.stack(episode, dim=0)
                batch.append(episode)
            batch = torch.stack(batch, dim=0)  # bs * n_cls * n_per
            yield batch.view(-1)
