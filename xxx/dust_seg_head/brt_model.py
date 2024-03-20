import os
import sys
import torch
import torchvision
from torch.nn import functional as F

import layers
# from dl.utils.colors import ModelType


class BrtResnetPyramidLite12(torch.nn.Module):
    """
    Network defining a Pixelseg BRT Residual Pyramid Network.
    More information on this architecture choice can be found in Confluence page:
    """
    def __init__(self, params):
        """
        param params: WorkFlowConfig params.
        """
        super().__init__()
#         self.modelType = ModelType.SEGMENTATION
        track_running_stats = True
        in_channels = params['input_dims']
        self.num_classes = params['num_classes']
        self.model_params = params['model_params']
        num_block_layers = self.model_params['num_block_layers']
        widening_factor = self.model_params['widening_factor']
        upsample_mode = self.model_params['upsample_mode']
        activation_fn = F.relu
        self.seg_output = params.get('seg_output', False)
        self.cls_output = params.get('cls_output', True)
        self.add_softmax_layer = params.get('add_softmax_layer', False)

        self.conv1 = layers.ConvBatchNormBlock(
                in_channels=in_channels,
                out_channels=widening_factor * 32,
                activation_fn=layers.null_activation_fn,
                is_track_running_stats=track_running_stats,
                kernel_size=5,
                stride=2,
                padding=2,
                name='conv_bn_block_1')

        self.pool1 = torch.nn.MaxPool2d(
                kernel_size=2,
                stride=2,
                return_indices=False)

        self.res_block_2a = layers.ResidualBlockOriginal(
                num_block_layers=num_block_layers,
                in_channels=widening_factor * 32,
                filters=[widening_factor * 32, widening_factor * 32],
                activation_fn=activation_fn,
                kernel_sizes=[3, 3],
                strides=[1, 1],
                dilation_rates=[1, 1],
                paddings=[1, 1],
                is_track_running_stats=track_running_stats,
                name='res_block_2a')

        self.res_block_2b = layers.ResidualBlockOriginal(
                num_block_layers=num_block_layers,
                in_channels=widening_factor * 32,
                filters=[widening_factor * 32, widening_factor * 32],
                activation_fn=activation_fn,
                kernel_sizes=[3, 3],
                strides=[1, 1],
                dilation_rates=[1, 1],
                paddings=[1, 1],
                is_track_running_stats=track_running_stats,
                name='res_block_2b')

        self.res_block_3a = layers.ResidualBlockOriginal(
                num_block_layers=num_block_layers,
                in_channels=widening_factor * 32,
                filters=[widening_factor * 64, widening_factor * 64],
                activation_fn=activation_fn,
                kernel_sizes=[5, 3],
                strides=[2, 1],
                dilation_rates=[1, 1],
                paddings=[2, 1],
                skip_conv_kernel_size=5,
                skip_conv_stride=2,
                skip_conv_dilation=1,
                skip_conv_padding=2,
                is_track_running_stats=track_running_stats,
                name='res_block_3a')

        self.res_block_4a = layers.ResidualBlockOriginal(
                num_block_layers=num_block_layers,
                in_channels=widening_factor * 64,
                filters=[widening_factor * 128, widening_factor * 128],
                activation_fn=activation_fn,
                kernel_sizes=[5, 3],
                strides=[2, 1],
                dilation_rates=[1, 1],
                paddings=[2, 1],
                skip_conv_kernel_size=5,
                skip_conv_stride=2,
                skip_conv_dilation=1,
                skip_conv_padding=2,
                is_track_running_stats=track_running_stats,
                name='res_block_4a')

        self.res_block_5a = layers.ResidualBlockOriginal(
                num_block_layers=num_block_layers,
                in_channels=widening_factor * 128,
                filters=[widening_factor * 128, widening_factor * 128],
                activation_fn=activation_fn,
                kernel_sizes=[5, 3],
                strides=[2, 1],
                dilation_rates=[1, 1],
                paddings=[2, 1],
                skip_conv_kernel_size=5,
                skip_conv_stride=2,
                skip_conv_dilation=1,
                skip_conv_padding=2,
                is_track_running_stats=track_running_stats,
                name='res_block_5a')
        
        if self.seg_output:
            # segmentation head
            self.unpool6 = layers.Interpolate(scale_factor=2, mode=upsample_mode)
            self.conv6 = torch.nn.Conv2d(
                    in_channels=int(widening_factor * 128),
                    out_channels=int(widening_factor * 128),
                    kernel_size=1,
                    stride=1,
                    padding=0)
            torch.nn.init.xavier_uniform_(self.conv6.weight)

            # ele_add_6 (256 conv6(res_block_4a) + unpool6(res_block_5a))
            self.conv_bn_6_7 = layers.ConvBatchNormBlock(
                    in_channels=int(widening_factor * 128),
                    out_channels=int(widening_factor * 64),
                    is_track_running_stats=track_running_stats,
                    kernel_size=3,
                    stride=1,
                    activation_fn=activation_fn,
                    padding=1,
                    name='conv_bn_6_7')

            self.unpool7 = layers.Interpolate(scale_factor=2, mode=upsample_mode)
            self.conv7 = torch.nn.Conv2d(
                    in_channels=int(widening_factor * 64),
                    out_channels=int(widening_factor * 64),
                    kernel_size=1,
                    stride=1,
                    padding=0)
            torch.nn.init.xavier_uniform_(self.conv7.weight)

            # ele_add_7 (128 conv7(res_block_3a) + unpool7(conv_bn_6_7(ele_add_6))
            self.conv_bn_7_8 = layers.ConvBatchNormBlock(
                in_channels=int(widening_factor * 64),
                out_channels=int(widening_factor * 32),
                is_track_running_stats=track_running_stats,
                kernel_size=3,
                stride=1,
                activation_fn=activation_fn,
                padding=1,
                name='conv_bn_7_8')

            self.unpool8 = layers.Interpolate(scale_factor=2, mode=upsample_mode)
            self.conv8 = torch.nn.Conv2d(
                in_channels=int(widening_factor * 32),
                out_channels=int(widening_factor * 32),
                kernel_size=1,
                stride=1,
                padding=0)
            torch.nn.init.xavier_uniform_(self.conv8.weight)

            # ele_add_8 (64 conv8(res_block_2b) + unpool8(conv_bn_7_8(ele_add_8))
            self.conv_bn_8_9 = layers.ConvBatchNormBlock(
                in_channels=int(widening_factor * 32),
                out_channels=int(widening_factor * 32),
                is_track_running_stats=track_running_stats,
                kernel_size=3,
                stride=1,
                activation_fn=activation_fn,
                padding=1,
                name='conv_bn_8_9')

            self.unpool9 = layers.Interpolate(scale_factor=2, mode=upsample_mode)
            # concat9 (64 + 64 conv1)
            self.conv9 = layers.ConvBatchNormBlock(
                    in_channels=int(widening_factor * 64),
                    out_channels=int(widening_factor * 32),
                    is_track_running_stats=track_running_stats,
                    kernel_size=3,
                    stride=1,
                    activation_fn=activation_fn,
                    padding=1,
                    name='conv_bn_block_9')

            # self.unpool10 = layers.Interpolate(scale_factor=2, mode=upsample_mode)
            self.conv10 = layers.ConvBatchNormBlock(
                    in_channels=int(widening_factor * 32),
                    out_channels=int(widening_factor * 16),
                    is_track_running_stats=track_running_stats,
                    kernel_size=3,
                    stride=1,
                    activation_fn=activation_fn,
                    padding=1,
                    name='conv_bn_block_10')

            # self.conv11 = layers.ConvBatchNormBlock(
            #         in_channels=int(widening_factor * 16),
            #         out_channels=int(widening_factor * 16),
            #         activation_fn=activation_fn,
            #         is_track_running_stats=track_running_stats,
            #         kernel_size=3,
            #         stride=1,
            #         padding=1,
            #         name='conv_bn_block_11')

            self.conv12 = layers.ConvBatchNormBlock(
                    in_channels=int(widening_factor * 16),
                    out_channels=int(widening_factor * 8),
                    activation_fn=activation_fn,
                    is_track_running_stats=track_running_stats,
                    kernel_size=7,
                    stride=1,
                    padding=3,
                    name='conv_bn_block_12')

            self.conv13 = torch.nn.Conv2d(
                in_channels=int(widening_factor * 8),
                out_channels=self.num_classes,
                kernel_size=1,
                stride=1,
                padding=0)
            torch.nn.init.xavier_uniform_(self.conv13.weight)

            self.unpool_logits = layers.Interpolate(scale_factor=2, mode=upsample_mode)
            # end of segmentation head
    
        if self.cls_output:
            # extension for plug classification
            self.res_block_6a = layers.ResidualBlockOriginal(
                    num_block_layers=num_block_layers,
                    in_channels=widening_factor * 128,
                    filters=[widening_factor * 48, widening_factor * 48],
                    activation_fn=activation_fn,
                    kernel_sizes=[5, 3],
                    strides=[2, 1],
                    dilation_rates=[1, 1],
                    paddings=[2, 1],
                    skip_conv_kernel_size=5,
                    skip_conv_stride=2,
                    skip_conv_dilation=1,
                    skip_conv_padding=2,
                    is_track_running_stats=track_running_stats,
                    name='res_block_6a')

            self.res_block_7a = layers.ResidualBlockOriginal(
                    num_block_layers=num_block_layers,
                    in_channels=widening_factor * 48,
                    filters=[widening_factor * 16, widening_factor * 16],
                    activation_fn=activation_fn,
                    kernel_sizes=[5, 3],
                    strides=[2, 1],
                    dilation_rates=[1, 1],
                    paddings=[2, 1],
                    skip_conv_kernel_size=5,
                    skip_conv_stride=2,
                    skip_conv_dilation=1,
                    skip_conv_padding=2,
                    is_track_running_stats=track_running_stats,
                    name='res_block_7a')

            self.pool2 = torch.nn.AvgPool2d(
                    kernel_size=[4, 8],
                    stride=[1, 1])

            self.fc1 = torch.nn.Linear(widening_factor * 16, 2)
            # end of extension


    def forward(self, x):
        # encoder
        conv1 = self.conv1(x)
        pool1 = self.pool1(conv1)
        res_block_2a = self.res_block_2a(pool1)
        res_block_2b = self.res_block_2b(res_block_2a)
        res_block_3a = self.res_block_3a(res_block_2b)
        res_block_4a = self.res_block_4a(res_block_3a)
        res_block_5a = self.res_block_5a(res_block_4a)

        if self.seg_output:
            # decoder
            unpool6 = self.unpool6(res_block_5a)
            conv6 = self.conv6(res_block_4a)
            ele_add_6 = unpool6 + conv6
            conv_bn_6_7 = self.conv_bn_6_7(ele_add_6)
            unpool7 = self.unpool7(conv_bn_6_7)
            conv7 = self.conv7(res_block_3a)
            ele_add_7 = unpool7 + conv7
            conv_bn_7_8 = self.conv_bn_7_8(ele_add_7)
            unpool8 = self.unpool8(conv_bn_7_8)
            conv8 = self.conv8(res_block_2b)
            ele_add_8 = unpool8 + conv8
            conv_bn_8_9 = self.conv_bn_8_9(ele_add_8)     # N, 64, 128, 256
            unpool9 = self.unpool9(conv_bn_8_9)           # N, 64, 256, 512
            concat9 = torch.cat((unpool9, conv1), dim=1)  # N, 128, 256, 512
            conv9 = self.conv9(concat9)                   # N, 64, 256, 512
            # unpool10 = self.unpool10(conv9)
            conv10 = self.conv10(conv9)
            # conv11 = self.conv11(conv10)
            conv12 = self.conv12(conv10)
            conv13 = self.conv13(conv12)
            logits = self.unpool_logits(conv13)   
        
        if self.cls_output:
            # output plug prediction
            # res_block_5a = res_block_5a[:, :, 1:9, 8:24]
            res_block_6a = self.res_block_6a(res_block_5a)
            res_block_7a = self.res_block_7a(res_block_6a)
            pool2 = self.pool2(res_block_7a)
            # pool2_squeezed = torch.squeeze(pool2)  # not supported
            pool2_squeezed = pool2[:, :, 0, 0]
            cls_logits = self.fc1(pool2_squeezed)

        if self.seg_output and self.cls_output and self.add_softmax_layer:
            softmax_logits = F.softmax(logits, dim=1)
            class_confidence, class_labels = torch.max(softmax_logits, dim=1, keepdim=True)
            return class_labels.to(torch.int32), class_confidence.to(torch.float32), cls_logits
        elif self.seg_output and self.add_softmax_layer:
            softmax_logits = F.softmax(logits, dim=1)
            # class_confidence, class_labels = torch.max(softmax_logits, dim=1, keepdim=True)
            # return class_labels.to(torch.int32), class_confidence.to(torch.float32)
            class_confidence, class_labels = torch.max(softmax_logits, dim=1, keepdim=False)
            return class_labels.to(torch.int32), softmax_logits.to(torch.float32)
        elif self.seg_output:
            return logits
        elif self.cls_output:
            return cls_logits
        else:
            print('WRONG CONFIG')
            sys.exit(1)


def freeze_encoder(model):
    for name, child in model.named_children():
        if not name in ['res_block_6a', 'res_block_7a', 'pool2', 'fc1']:
            for param in child.parameters():
                param.requires_grad = False
    return model


def load_states(model, snapshot_path):
    snapshot = torch.load(snapshot_path, map_location=lambda storage, loc: storage)
    STATE_DICT_KEY = 'state_dict'
    model.load_state_dict(snapshot[STATE_DICT_KEY], strict=False)
    return model


if __name__ == '__main__':
    # build model
    params = {"input_dims": 3, "num_classes": 2, "seg_output": True, "cls_output": False, "add_softmax_layer": True,
              "model_params": {"num_block_layers": 2, "widening_factor": 2, "upsample_mode": "nearest"}}
    model = BrtResnetPyramidLite12(params)
    print('# tunable params:', sum(p.numel() for p in model.parameters() if p.requires_grad))
    
    # load weights
    weight_file = '/home/bluerivertech/li.yu/exps/driveable_terrain_model/v471_rd_2cls_dustseghead_0808/job_quality_val_bestmodel.pth'
    model = load_states(model, weight_file)
    
    # check input and output
    dummy_input = torch.randn((1, 3, 512, 1024))
    label, conf = model(dummy_input)
    print(label.shape, conf.shape)
    