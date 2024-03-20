#!/usr/bin/env python3.6

from collections import OrderedDict
from typing import AnyStr, Iterator, Tuple

import torch
from torch.nn import functional as F


BATCH_NORM_EPSILON = 1e-5


class ConvBatchNormBlock(torch.nn.Module):
    def __init__(self,
                in_channels : int,
                out_channels : int,
                is_track_running_stats : bool,
                kernel_size : int,
                stride : int,
                activation_fn : torch.nn.Module = None,
                padding : int = 0,
                dilation : int = 1,
                groups : int = 1,
                bias : bool = True,
                bn_momentum : float = 0.1,
                bn_affine : bool = True,
                name : AnyStr = ''):
        """
        https://pytorch.org/docs/stable/nn.html#convolution-layers
        https://pytorch.org/docs/stable/nn.html?highlight=batch%20norm#torch.nn.BatchNorm2d
        """
        super().__init__()

        self.name = name
        self.conv_layer = torch.nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=True)
        self.batch_norm_layer = torch.nn.BatchNorm2d(
            num_features=out_channels,
            eps=BATCH_NORM_EPSILON,
            momentum=bn_momentum,
            affine=bn_affine,
            track_running_stats=is_track_running_stats)
        if activation_fn is None:
            self.activation_fn = null_activation_fn
        else:
            self.activation_fn = activation_fn
        self._initialize_weights()

    def _initialize_weights(self):
        torch.nn.init.xavier_uniform_(self.conv_layer.weight)

    def forward(self, inputs):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        conv_tensor = self.conv_layer(inputs)
        bn_tensor = self.batch_norm_layer(conv_tensor)
        return self.activation_fn(bn_tensor)


class ResidualBlockOriginal(torch.nn.Module):
    def __init__(self,
                num_block_layers : int,
                in_channels : int,
                filters : Iterator,
                activation_fn : torch.nn.Module,
                kernel_sizes : Iterator,
                strides : Iterator,
                dilation_rates : Iterator,
                paddings : Iterator,
                skip_conv_kernel_size : int = None,
                skip_conv_stride : int = None,
                skip_conv_dilation : int = None,
                skip_conv_padding : int = None,
                is_track_running_stats : bool = True,
                name : AnyStr = ''):

        super().__init__()
        self.name = name
        self.num_block_layers = num_block_layers
        self.filters = filters
        self.activation_fn = activation_fn

        # Conv params
        self.kernel_sizes = kernel_sizes
        self.strides = strides
        self.dilation_rates = dilation_rates
        self.paddings = paddings

        if len(filters) != num_block_layers:
            raise ValueError('filters array must have num_layers elements.')
        if len(kernel_sizes) != num_block_layers:
            raise ValueError('kernel_sizes array must have num_layers elements.')
        if len(strides) != num_block_layers:
            raise ValueError('strides array must have num_layers elements.')
        if len(dilation_rates) != num_block_layers:
            raise ValueError('dilation_rates array must have num_layers elements.')
        if len(paddings) != num_block_layers:
            raise ValueError('paddings array must have num_layers elements.')

        layers_dict = OrderedDict()
        current_in_channels = in_channels
        for i in range(self.num_block_layers):
            layers_dict['conv_{}'.format(i)] = torch.nn.Conv2d(
                in_channels=current_in_channels,
                out_channels=filters[i],
                kernel_size=kernel_sizes[i],
                stride=strides[i],
                padding=paddings[i],
                dilation=dilation_rates[i],
                groups=1,
                bias=True)

            layers_dict['batch_norm_{}'.format(i)] = torch.nn.BatchNorm2d(
                num_features=filters[i],
                eps=BATCH_NORM_EPSILON,
                momentum=0.1,
                affine=True,
                track_running_stats=is_track_running_stats)
            current_in_channels = filters[i]


        if all(x is not None for x in
                [skip_conv_kernel_size, skip_conv_stride, skip_conv_dilation, skip_conv_padding]):
            layers_dict['skip_connection_conv'] = torch.nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=filters[-1],
                    kernel_size=skip_conv_kernel_size,
                    stride=skip_conv_stride,
                    padding=skip_conv_padding,
                    dilation=skip_conv_dilation,
                    bias=True)
            layers_dict['skip_connection_bn'] = torch.nn.BatchNorm2d(
                num_features=filters[-1],
                eps=BATCH_NORM_EPSILON,
                momentum=0.1,
                affine=True,
                track_running_stats=is_track_running_stats)

        # Init layers in nn.Sequential wrapper with layer order preserved from OrderedDict.
        self.sequential_layers = torch.nn.Sequential(layers_dict)
        self._initialize_weights()

    def _initialize_weights(self):
        for name, layer in self.sequential_layers.named_children():
            if 'conv' in name:
                torch.nn.init.xavier_uniform_(layer.weight)

    def forward(self, input_tensor):
        identity_tensor = input_tensor
        residual_tensor = input_tensor
        for i in range(self.num_block_layers):
            if not (hasattr(self.sequential_layers, 'conv_{}'.format(i)) and
                    hasattr(self.sequential_layers, 'batch_norm_{}'.format(i))):
                raise LookupError('Could not find conv and batch_norm layers')

            conv_layer = getattr(self.sequential_layers, 'conv_{}'.format(i))
            bn_layer = getattr(self.sequential_layers, 'batch_norm_{}'.format(i))

            residual_tensor = conv_layer(residual_tensor)
            residual_tensor = bn_layer(residual_tensor)

            # Do not attach a activation to last conv layer before residual connection.
            if i < (self.num_block_layers - 1):
                residual_tensor = self.activation_fn(residual_tensor)

        # Extra conv layer to increase input dimension to match with output dimension.
        if not (hasattr(self.sequential_layers, 'skip_connection_conv') and
                hasattr(self.sequential_layers, 'skip_connection_bn')):
            eltwise_add_tensor = residual_tensor + identity_tensor
            return self.activation_fn(eltwise_add_tensor)
        else:
            skip_conv_layer = getattr(self.sequential_layers, 'skip_connection_conv')
            skip_bn_layer = getattr(self.sequential_layers, 'skip_connection_bn')

            skip_conv_tensor = skip_conv_layer(identity_tensor)
            skip_bn_tensor = skip_bn_layer(skip_conv_tensor)
            eltwise_add_tensor = residual_tensor + skip_bn_tensor
            return self.activation_fn(eltwise_add_tensor)


class Interpolate(torch.nn.Module):
    """
    For some reason torch doesn't have a class version of this functional:
    https://pytorch.org/docs/stable/nn.html#torch.nn.functional.interpolate
    """
    def __init__(self, size : Tuple = None,
                 scale_factor : float = None,
                 mode : AnyStr = 'nearest',
                 align_corners : bool = None):
        super().__init__()
        self.size = size
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x):
        return F.interpolate(x, size=self.size, scale_factor=self.scale_factor,
            mode=self.mode, align_corners=self.align_corners)


"""
Activation functions:
"""
class Activation(torch.nn.Module):
    def __init__(self, activation_fn):
        super().__init__()
        self.activation_fn = activation_fn

    def forward(self, x):
        return self.activation_fn(x)


def swish_fn(x, beta=1.0):
    return x * torch.sigmoid(x * beta)


def null_activation_fn(x):
    return x


def make_one_hot(labels, num_classes, ignore_label):
    """
    Converts an integer label torch.autograd.Variable to a one-hot Variable.
    This function also makes the ignore label to be num_classes + 1 index and slices it from target.


    :labels : torch.autograd.Variable of torch.cuda.LongTensor N x 1 x H x W, where N is batch size.
        Each value is an integer representing correct classification.
    :param num_classes: Number of classes in labels.
    :return target : torch.autograd.Variable of torch.cuda.FloatTensor
    N x C x H x W, where C is class number. One-hot encoded.
    """
    labels = labels.long()
    mask = (labels == ignore_label)
    labels[mask] = num_classes
    one_hot = torch.cuda.FloatTensor(labels.size(0), num_classes+1, labels.size(2), labels.size(3)).zero_()
    target = one_hot.scatter_(1, labels.data, 1)
    target = target[:, :num_classes, :, :]
    target = torch.autograd.Variable(target)
    labels[mask] = ignore_label
    return target
