# coding=utf-8
# Copyright (c) DIRECT Contributors

# Code borrowed and edited from https://github.com/lpj-github-io/MWCNNv2/blob/master/MWCNN_code/model/.

import torch.nn as nn
import torch.nn.functional as F

from direct.nn.mwcnn.common import DWT, IWT, BBlock, DBlock_com, DBlock_com1, DBlock_inv, DBlock_inv1, default_conv



class MWCNN(nn.Module):
    def __init__(self, input_channels, num_hidden_channels, act=nn.ReLU(True), bn=False):
        super(MWCNN, self).__init__()

        self._kernel_size = 3

        self.DWT = DWT()
        self.IWT = IWT()

        conv = default_conv

        m_head = [BBlock(conv, input_channels, num_hidden_channels, self._kernel_size, act=act)]

        d_l0 = [DBlock_com1(conv, num_hidden_channels, num_hidden_channels, self._kernel_size, act=act, bn=bn)]

        d_l1 = list()
        d_l1.append(BBlock(conv, num_hidden_channels * 4, num_hidden_channels * 2, self._kernel_size, act=act, bn=bn))
        d_l1.append(DBlock_com1(conv, num_hidden_channels * 2, num_hidden_channels * 2, self._kernel_size, act=act, bn=bn))

        d_l2 = list()
        d_l2.append(BBlock(conv, num_hidden_channels * 8, num_hidden_channels * 4, self._kernel_size, act=act, bn=bn))
        d_l2.append(DBlock_com1(conv, num_hidden_channels * 4, num_hidden_channels * 4, self._kernel_size, act=act, bn=bn))

        pro_l3 = list()
        pro_l3.append(BBlock(conv, num_hidden_channels * 16, num_hidden_channels * 8, self._kernel_size, act=act, bn=bn))
        pro_l3.append(DBlock_com(conv, num_hidden_channels * 8, num_hidden_channels * 8, self._kernel_size, act=act, bn=bn))
        pro_l3.append(DBlock_inv(conv, num_hidden_channels * 8, num_hidden_channels * 8, self._kernel_size, act=act, bn=bn))
        pro_l3.append(BBlock(conv, num_hidden_channels * 8, num_hidden_channels * 16, self._kernel_size, act=act, bn=bn))

        i_l2 = list()
        i_l2.append(DBlock_inv1(conv, num_hidden_channels * 4, num_hidden_channels * 4, self._kernel_size, act=act, bn=bn))
        i_l2.append(BBlock(conv, num_hidden_channels * 4, num_hidden_channels * 8, self._kernel_size, act=act, bn=bn))

        i_l1 = list()
        i_l1.append(DBlock_inv1(conv, num_hidden_channels * 2, num_hidden_channels * 2, self._kernel_size, act=act, bn=bn))
        i_l1.append(BBlock(conv, num_hidden_channels * 2, num_hidden_channels * 4, self._kernel_size, act=act, bn=bn))

        i_l0 = [DBlock_inv1(conv, num_hidden_channels, num_hidden_channels, self._kernel_size, act=act, bn=bn)]

        m_tail = [conv(num_hidden_channels, input_channels, self._kernel_size)]

        self.head = nn.Sequential(*m_head)
        self.d_l2 = nn.Sequential(*d_l2)
        self.d_l1 = nn.Sequential(*d_l1)
        self.d_l0 = nn.Sequential(*d_l0)
        self.pro_l3 = nn.Sequential(*pro_l3)
        self.i_l2 = nn.Sequential(*i_l2)
        self.i_l1 = nn.Sequential(*i_l1)
        self.i_l0 = nn.Sequential(*i_l0)
        self.tail = nn.Sequential(*m_tail)

    @staticmethod
    def pad(x):
        padding = [0, 0, 0, 0]

        if x.shape[-2] % 2 != 0:
            padding[3] = 1  # Padding right - width
        if x.shape[-1] % 2 != 0:
            padding[1] = 1  # Padding bottom - height
        if sum(padding) != 0:
            x = F.pad(x, padding, "reflect")

        return x

    @staticmethod
    def crop_to_shape(x, shape):
        h, w = x.shape[-2:]

        if h > shape[0]:
            x = x[:, :, :shape[0], :]
        if w > shape[1]:
            x = x[:, :, :, :shape[1]]

        return x


    def forward(self, x):

        x0 = self.pad(self.d_l0(self.head(self.pad(x.clone()))))
        x1 = self.pad(self.d_l1(self.DWT(x0)))
        x2 = self.pad(self.d_l2(self.DWT(x1)))

        y = self.crop_to_shape(self.IWT(self.pro_l3(self.DWT(x2))), x2.shape[-2:]) + x2
        y = self.crop_to_shape(self.IWT(self.i_l2(y)), x1.shape[-2:]) + x1
        y = self.crop_to_shape(self.IWT(self.i_l1(y)), x0.shape[-2:]) + x0

        x = self.crop_to_shape(self.tail(self.i_l0(y)), x.shape[-2:]) + x

        return x
