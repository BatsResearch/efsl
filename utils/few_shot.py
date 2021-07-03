import torch
import utils
import torch.nn.functional as F



def split_shot_query(data, way, shot, query, pseudo, ep_per_batch=1):
    img_shape = data.shape[1:]
    data = data.view(ep_per_batch, way, shot + query + pseudo, *img_shape)
    x_shot, x_query, x_pseudo = data.split([shot, query, pseudo], dim=2)
    x_shot = x_shot.contiguous()
    x_pseudo = x_pseudo.contiguous()
    x_query = x_query.contiguous().view(ep_per_batch, way * query, *img_shape)
    return x_shot, x_query, x_pseudo


def make_nk_label(n, k, ep_per_batch=1):
    label = torch.arange(n).unsqueeze(1).expand(n, k).reshape(-1)
    label = label.repeat(ep_per_batch)
    return label


def predict(model, data, n_way, n_shot, n_query, n_pseudo, ep_per_batch, return_log=False):
    x_shot, x_query, x_pseudo = split_shot_query(
        data=data.cuda(),
        way=n_way,
        shot=n_shot,
        query=n_query,
        pseudo=n_pseudo,
        ep_per_batch=ep_per_batch)

    label = make_nk_label(n=n_way, k=n_query,
                          ep_per_batch=ep_per_batch).cuda()

    logits, raw_log = model(x_shot=x_shot,
                            x_query=x_query,
                            x_pseudo=x_pseudo,
                            return_log=return_log
                            )
    logits = logits.view(-1, n_way)
    loss = F.cross_entropy(logits, label)
    acc = utils.compute_acc(logits, label)

    return logits, acc, loss, raw_log
