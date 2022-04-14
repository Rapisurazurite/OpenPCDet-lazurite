# -*- coding: utf-8 -*-
# @Time    : 4/14/2022 2:51 PM
# @Author  : Lazurite
# @Email   : lazurite@tom.com
# @File    : SSFA_bakebone.py.py
# @Software: PyCharm


import numpy as np
import torch
import torch.nn as nn


class SSFA(nn.Module):
    """
    BACKBONE_2D:
        NAME:SSFA
        INPUT_CHANNELS:input_channel

    """

    def __init__(self, model_cfg, input_channels):
        super().__init__()
        self.model_cfg = model_cfg
        input_channels = self.model_cfg['INPUT_CHANNELS']
        self.input_channels = input_channels


        # [input_channels, 200, 176] -> [input_channels, 200, 176]
        self.bottom_up_block_0 = nn.Sequential(
            nn.Conv2d(input_channels, input_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(input_channels, eps=1e-3, momentum=0.01),
            nn.ReLU(),

            nn.Conv2d(input_channels, input_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(input_channels, eps=1e-3, momentum=0.01),
            nn.ReLU(),

            nn.Conv2d(input_channels, input_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(input_channels, eps=1e-3, momentum=0.01),
            nn.ReLU()
        )

        sementic_group_channel = input_channels * 2
        # [input_channels, 200, 176] -> [sementic_group_channel, 100, 88]
        self.bottom_up_block_1 = nn.Sequential(
            nn.Conv2d(input_channels, sementic_group_channel, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(sementic_group_channel, eps=1e-3, momentum=0.01),
            nn.ReLU(),

            nn.Conv2d(2 * input_channels, sementic_group_channel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(sementic_group_channel, eps=1e-3, momentum=0.01),
            nn.ReLU(),

            nn.Conv2d(2 * input_channels, sementic_group_channel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(sementic_group_channel, eps=1e-3, momentum=0.01),
            nn.ReLU()
        )

        self.trans_0 = nn.Sequential(
            nn.Conv2d(input_channels, input_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(input_channels, eps=1e-3, momentum=0.01),
            nn.ReLU()
        )

        self.trans_1 = nn.Sequential(
            nn.Conv2d(sementic_group_channel, sementic_group_channel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(sementic_group_channel, eps=1e-3, momentum=0.01),
            nn.ReLU()
        )

        self.deconv_block_0 = nn.Sequential(
            nn.ConvTranspose2d(sementic_group_channel, input_channels, kernel_size=3, stride=2, padding=1,
                               output_padding=1, bias=False),
            nn.BatchNorm2d(input_channels, eps=1e-3, momentum=0.01),
            nn.ReLU()
        )

        self.deconv_block_1 = nn.Sequential(
            nn.ConvTranspose2d(sementic_group_channel, input_channels, kernel_size=3, stride=2, padding=1,
                               output_padding=1, bias=False),
            nn.BatchNorm2d(input_channels, eps=1e-3, momentum=0.01),
            nn.ReLU()
        )

        self.conv_0 = nn.Sequential(
            nn.Conv2d(input_channels, input_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(input_channels, eps=1e-3, momentum=0.01),
            nn.ReLU()
        )

        self.conv_1 = nn.Sequential(
            nn.Conv2d(input_channels, input_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(input_channels, eps=1e-3, momentum=0.01),
            nn.ReLU()
        )

        self.w_0 = nn.Sequential(
            nn.Conv2d(input_channels, 1, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(1, eps=1e-3, momentum=0.01)
        )

        self.w_1 = nn.Sequential(
            nn.Conv2d(input_channels, 1, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(1, eps=1e-3, momentum=0.01)
        )

        self.num_bev_features = input_channels

    # def init_weights(self):
    #     for m in self.modules():
    #         if isinstance(m, nn.Conv2d):
    #             # use xavier_uniform
    #             nn.init.xavier_uniform(m.weight)

    def forward(self, data_dict):
        """
        Args:
            data_dict:
                spatial_features
        Returns:
        """
        spatial_features = data_dict['spatial_features']
        assert spatial_features.shape[-3:] == (self.input_channels, 200, 176)

        x_0 = self.bottom_up_block_0(spatial_features) # [input_channel, 200, 176] -> [input_channel, 200, 176]
        x_1 = self.bottom_up_block_1(x_0) # [input_channel, 200, 176] -> [input_channel, 100, 88]
        x_tran_0 = self.trans_0(x_0) # [input_channel, 200, 176] -> [input_channel, 200, 176]
        x_tran_1 = self.trans_1(x_1) # [input_channel, 100, 88] -> [input_channel, 100, 88]

        x_middle_0 = self.deconv_block_0(x_tran_1) + x_tran_0 # [input_channel, 200, 176] -> [input_channel, 200, 176]
        x_middle_1 = self.deconv_block_1(x_tran_1) # [input_channel, 100, 88] -> [input_channel, 200, 176]

        x_output_0, x_output_1 = self.conv_0(x_middle_0), self.conv_1(x_middle_1) # [input_channel, 200, 176] -> [input_channel, 200, 176]
        x_weight_0, x_weight_1 = self.w_0(x_output_0), self.w_1(x_output_1) # [input_channel, 200, 176] -> [input_channel, 1, 1]
        x_weight = torch.softmax(torch.cat([x_weight_0, x_weight_1], dim=1), dim=1) # [input_channel, 2]

        x_out = x_output_0 * x_weight[:, 0:1, :, :] + x_output_1 * x_weight[:, 1:2, :, :] # [input_channel, 200, 176]

        data_dict['spatial_features_2d'] = x_out
        return data_dict
