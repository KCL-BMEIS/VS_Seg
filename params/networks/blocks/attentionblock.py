import torch
from params.networks.blocks.convolutions import Convolution
from monai.networks.layers.factories import Act


class AttentionBlock1(torch.nn.Module):
    def __init__(self, dimensions, in_channels, out_channels, kernel_size, norm, dropout):
        super(AttentionBlock1, self).__init__()
        self.in_channels = in_channels
        self.conv1 = Convolution(dimensions,
                                 in_channels,
                                 in_channels//2,
                                 strides=1,
                                 kernel_size=kernel_size,
                                 act=Act.RELU,
                                 norm=norm,
                                 dropout=None,
                                 )

        self.conv2 = Convolution(dimensions,
                                 in_channels//2,
                                 out_channels=1,
                                 strides=1,
                                 kernel_size=kernel_size,
                                 act=Act.SIGMOID,
                                 norm=norm,
                                 dropout=None,
                                 )

    def forward(self, x):
        att = self.conv1(x)
        att = self.conv2(att)
        return att, x


class AttentionBlock2(torch.nn.Module):
    def __init__(self, dimensions, in_channels, out_channels, kernel_size, norm, dropout):
        super(AttentionBlock2, self).__init__()
        self.in_channels = in_channels

    def forward(self, input_tuple):
        (att, x) = input_tuple
        att_tile = att.repeat([1, self.in_channels, 1, 1, 1]) * x
        output = att_tile + x
        return output
