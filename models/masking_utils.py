import torch
import torch.nn as nn
import torch.nn.functional as F

from .models import register
from .resnet_new import conv3x3, DropBlock, SELayer, BasicBlock


@register('mean-aggregator')
class MeanAggregator(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.mean(x, dim=-4)


class MultimoduleMasking(nn.Module):

    def __init__(self, maskingmodule, inplanes, channels, final_relu, activation, max_pool, **kwargs):
        super().__init__()

        self.masking_model = maskingmodule(inplanes=inplanes, channels=channels, final_relu=final_relu, max_pool=max_pool, **kwargs)
        self.out_dim = self.masking_model.out_dim
        if activation == 'sigmoid':
            self.act_func = nn.Sigmoid()
        elif activation == 'softmax':
            self.act_func = nn.Softmax(dim=0)
        elif activation == 'linear':
            self.act_func = None
        else:
            raise ValueError('Invalid Activation Function.')

    def forward(self, x):

        mask = self.masking_model(x)

        if self.act_func is not None:
            mask = self.act_func(mask)

        return mask


class Multimodule(nn.Module):
    def __init__(self, block, inplanes, channels, keep_prob=1.0, avg_pool=False, drop_rate=0.0,
                 dropblock_size=5, use_se=False, final_relu=True, max_pool=True):
        super(Multimodule, self).__init__()

        self.inplanes = inplanes
        self.use_se = use_se
        self.layers = nn.ModuleList()
        for channel in channels[:-1]:
            self.layers.append(self._make_layer(block, 1, channel,
                                       stride=2, drop_rate=drop_rate, drop_block=True, block_size=dropblock_size, max_pool=max_pool))
        self.layers.append(self._make_layer(block, 1, channels[-1],
                                   stride=2, drop_rate=drop_rate, drop_block=True, block_size=dropblock_size,
                                            final_relu=final_relu, max_pool=max_pool))
        self.out_dim = channels[-1]
        # if avg_pool:
            # self.avgpool = nn.AvgPool2d(5, stride=1)
            # self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.keep_prob = keep_prob
        # self.keep_avg_pool = avg_pool
        self.dropout = nn.Dropout(p=1 - self.keep_prob, inplace=False)
        self.drop_rate = drop_rate

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, n_block, planes, stride=1, drop_rate=0.0, drop_block=False, block_size=1, final_relu=True, max_pool=True):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        if n_block == 1:
            layer = block(self.inplanes, planes, stride, downsample, drop_rate, drop_block, block_size, self.use_se, final_relu=final_relu, max_pool=max_pool)
        else:
            layer = block(self.inplanes, planes, stride, downsample, drop_rate, self.use_se, final_relu=final_relu, max_pool=max_pool)
        layers.append(layer)
        self.inplanes = planes * block.expansion

        for i in range(1, n_block):
            if i == n_block - 1:
                layer = block(self.inplanes, planes, drop_rate=drop_rate, drop_block=drop_block,
                              block_size=block_size, use_se=self.use_se, final_relu=final_relu, max_pool=max_pool)
            else:
                layer = block(self.inplanes, planes, drop_rate=drop_rate, use_se=self.use_se, final_relu=final_relu, max_pool=max_pool)
            layers.append(layer)

        return nn.Sequential(*layers)

    def forward(self, x):

        for layer in self.layers:
            x = layer(x)

        return x


class BasicLayer(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, drop_rate=0.0, drop_block=False,
                 block_size=1, use_se=False, final_relu=True, max_pool=True, residual=True):
        super(BasicLayer, self).__init__()
        self.conv1 = conv3x3(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.LeakyReLU(0.1)
        # self.conv2 = conv3x3(planes, planes)
        # self.bn2 = nn.BatchNorm2d(planes)
        # self.conv3 = conv3x3(planes, planes)
        # self.bn3 = nn.BatchNorm2d(planes)
        if max_pool:
            self.maxpool = nn.MaxPool2d(stride)
        else:
            self.maxpool = None
        self.final_relu = final_relu
        self.residual = residual
        self.downsample = downsample
        self.stride = stride
        self.drop_rate = drop_rate
        self.num_batches_tracked = 0
        self.drop_block = drop_block
        self.block_size = block_size
        self.DropBlock = DropBlock(block_size=self.block_size)
        self.use_se = use_se
        if self.use_se:
            self.se = SELayer(planes, 4)

    def forward(self, x):
        self.num_batches_tracked += 1

        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        # out = self.relu(out)

        # out = self.conv2(out)
        # out = self.bn2(out)
        # out = self.relu(out)
        #
        # out = self.conv3(out)
        # out = self.bn3(out)
        if self.use_se:
            out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        if self.residual:
            out += residual
        if self.final_relu:
            out = self.relu(out)
        if self.maxpool:
            out = self.maxpool(out)

        if self.drop_rate > 0:
            if self.drop_block == True:
                feat_size = out.size()[2]
                keep_rate = max(1.0 - self.drop_rate / (20 * 2000) * (self.num_batches_tracked), 1.0 - self.drop_rate)
                gamma = (1 - keep_rate) / self.block_size ** 2 * feat_size ** 2 / (feat_size - self.block_size + 1) ** 2
                out = self.DropBlock(out, gamma=gamma)
            else:
                out = F.dropout(out, p=self.drop_rate, training=self.training, inplace=True)

        return out


@register('multi-block')
def MultiBlock(**kwargs):
    return Multimodule(BasicBlock, **kwargs)


@register('multi-layer')
def MultiLayer(**kwargs):
    return Multimodule(BasicLayer, **kwargs)


@register('multi-block-masking')
def MultiBlockMasking(**kwargs):
    return MultimoduleMasking(MultiBlock, **kwargs)


@register('multi-layer-masking')
def MultiLayerMasking(**kwargs):
    return MultimoduleMasking(MultiLayer, **kwargs)


def encoder_wrapper(encoder_name, encoder, x_shot, x_pseudo, x_query, **kwargs):

    if 'resnet' not in encoder_name:
        x_shot, x_pseudo, x_query = encoder(x_shot, x_pseudo, x_query)
    else:
        shot_shape = x_shot.shape[:-3]
        pseudo_shape = x_pseudo.shape[:-3]
        query_shape = x_query.shape[:-3]
        img_shape = x_shot.shape[-3:]

        x_shot = x_shot.view(-1, *img_shape)  # [n_episodes*n_way*n_shot, 3, 84, 84]
        x_pseudo = x_pseudo.view(-1, *img_shape)  # [n_pseudo*n_way*n_pseudo, 3, 84, 84]
        x_query = x_query.view(-1, *img_shape)  # [n_episodes*n_way*n_query, 3, 84, 84]

        x_tot = encoder(torch.cat([x_shot, x_pseudo, x_query], dim=0))  # [n_episodes*n_way*(n_shot+n_pseudo+n_query), masking_channels, dropblock_sz, dropblock_sz]
        x_shot, x_pseudo, x_query = x_tot[:len(x_shot)], x_tot[len(x_shot):len(x_shot) + len(x_pseudo)], x_tot[len(
            x_shot) + len(x_pseudo):]

        x_shot = x_shot.view(*shot_shape, *x_tot.shape[
                                           1:])  # [n_episodes, n_way, n_shot, masking_channels, dropblock_sz, dropblock_size]
        x_pseudo = x_pseudo.view(*pseudo_shape, *x_tot.shape[1:])
        x_query = x_query.view(*query_shape, *x_tot.shape[1:])

    return x_shot, x_pseudo, x_query


if __name__ == '__main__':
    x = torch.randn(7, 512, 5, 5).cuda()
    multiblock = MultiLayer(inplanes=512, channels=[512, 111], final_relu=True, max_pool=1).cuda()
    y = multiblock(x)
    print(y.shape)
