"""Networks to test NGA.

Author: Yuhuang Hu
Email : yuhuang.hu@ini.uzh.ch
"""

from __future__ import print_function, absolute_import

import torch
from torch import nn

from evtrans.pyyolo.model import ConvLayer
from evtrans.pyyolo.model import make_conv_and_res_block
from evtrans.lenet import C1, C2


class LeNetFrontend(nn.Module):
    def __init__(self, input_dim=1):
        super(LeNetFrontend, self).__init__()

        self.c1 = C1(input_dim)
        self.c2_1 = C2()
        self.c2_2 = C2()

    def forward(self, img):

        output = self.c1(img)

        x = self.c2_1(output)
        output = self.c2_2(output)

        output += x

        return output


class LeNetFrameFrontend(nn.Module):
    def __init__(self, pretrained):
        super(LeNetFrameFrontend, self).__init__()

        self.c1 = pretrained.c1
        self.c2_1 = pretrained.c2_1
        self.c2_2 = pretrained.c2_2

    def forward(self, img):

        output = self.c1(img)

        x = self.c2_1(output)
        output = self.c2_2(output)

        output += x

        return output


class LeNetDetectNet(nn.Module):
    def __init__(self, pretrained):
        super(LeNetDetectNet, self).__init__()

        self.c3 = pretrained.c3
        self.f4 = pretrained.f4
        self.f5 = pretrained.f5

    def forward(self, x):
        output = x
        output = self.c3(output)
        output = nn.Flatten()(output)
        output = self.f4(output)
        output = self.f5(output)
        return output


class FrameYOLOFrontend(nn.Module):
    """The frame driven yolo frontend."""
    def __init__(self, pretrained_model, stage):
        """Specify number of stage from 1 to 3."""
        super(FrameYOLOFrontend, self).__init__()

        self.stage = stage

        # assume the pretrained model is the darkenet backbone.

        self.conv1 = pretrained_model.conv1

        self.blocks = nn.ModuleList()

        if stage >= 1:
            self.blocks.append(pretrained_model.cr_block1)
        if stage >= 2:
            self.blocks.append(pretrained_model.cr_block2)
        if stage >= 3:
            self.blocks.append(pretrained_model.cr_block3)

    def forward(self, x):
        tmp = self.conv1(x)

        for block in self.blocks:
            tmp = block(tmp)

        return tmp


class EVYOLOFrontend(nn.Module):
    """A Event driven YOLO Frontend."""
    def __init__(self, stage, input_dim=3):
        """Specify number of stage from 1 to 3."""
        super(EVYOLOFrontend, self).__init__()

        self.stage = stage

        self.conv1 = ConvLayer(input_dim, 32, 3)

        self.blocks = nn.ModuleList()

        if stage >= 1:
            self.blocks.append(
                make_conv_and_res_block(32, 64, 1))
        if stage >= 2:
            self.blocks.append(
                make_conv_and_res_block(64, 128, 2))
        if stage >= 3:
            self.blocks.append(
                make_conv_and_res_block(128, 256, 8))

    def forward(self, x):
        tmp = self.conv1(x)

        for block in self.blocks:
            tmp = block(tmp)

        return tmp


class YOLODetectNet(nn.Module):
    """The YOLO detection network."""
    def __init__(self, pretrained_model, cut_stage, nms=False, post=True):
        """Detection Net.

        the pretrained model consists of the entire yolo network.
        """
        super(YOLODetectNet, self).__init__()

        self.cut_stage = cut_stage

        self.dark_blocks = nn.ModuleList([
            getattr(pretrained_model.darknet, "cr_block{}".format(stage_i))
            for stage_i in range(self.cut_stage+1, 6)])

        self.pretrained_model = pretrained_model

    def darknet(self, x):
        """the input is from either frame or event driven frontend."""

        if self.cut_stage == 3:
            outputs = [x]
        else:
            outputs = []

        in_feature = x
        for block in self.dark_blocks:
            in_feature = block(in_feature)
            outputs.append(in_feature)

        return tuple(outputs[-3:])

    def forward(self, x):
        # reverse the order
        tmp3, tmp2, tmp1 = self.darknet(x)

        out1, out2, out3 = self.pretrained_model.yolo_tail(tmp1, tmp2, tmp3)
        out = torch.cat((out1, out2, out3), 1)
        return out

    def yolo_last_layers(self):
        _layers = [self.pretrained_model.yolo_tail.detect1.conv7,
                   self.pretrained_model.yolo_tail.detect2.conv7,
                   self.pretrained_model.yolo_tail.detect3.conv7]
        return _layers

    def yolo_last_two_layers(self):
        _layers = self.yolo_last_layers() + \
                  [self.pretrained_model.yolo_tail.detect1.conv6,
                   self.pretrained_model.yolo_tail.detect2.conv6,
                   self.pretrained_model.yolo_tail.detect3.conv6]
        return _layers

    def yolo_last_three_layers(self):
        _layers = self.yolo_last_two_layers() + \
                  [self.pretrained_model.yolo_tail.detect1.conv5,
                   self.pretrained_model.yolo_tail.detect2.conv5,
                   self.pretrained_model.yolo_tail.detect3.conv5]
        return _layers

    def yolo_tail_layers(self):
        _layers = [self.yolo_tail]
        return _layers

    def yolo_last_n_layers(self, n):
        try:
            n = int(n)
        except ValueError:
            pass
        if n == 1:
            return self.yolo_last_layers()
        elif n == 2:
            return self.yolo_last_two_layers()
        elif n == 3:
            return self.yolo_last_three_layers()
        elif n == 'tail':
            return self.yolo_tail_layers()
        else:
            raise ValueError("n>3 not defined")
