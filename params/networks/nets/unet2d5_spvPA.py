# Copyright 2020 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch.nn as nn

from params.networks.blocks.convolutions import Convolution, ResidualUnit
from ..blocks.attentionblock import AttentionBlock1, AttentionBlock2
from monai.networks.layers.factories import Norm, Act
from monai.networks.layers.simplelayers import SkipConnection
from monai.utils import export
from monai.utils.aliases import alias

@export("monai.networks.nets")
@alias("Unet2d5_spvPA")
class UNet2d5_spvPA(nn.Module):
    def __init__(
            self,
            dimensions,
            in_channels,
            out_channels,
            channels,
            strides,
            kernel_sizes,
            sample_kernel_sizes,
            num_res_units=0,
            act=Act.PRELU,
            norm=Norm.INSTANCE,
            dropout=0,
            attention_module=True,
    ):
        super().__init__()
        assert len(channels) == len(kernel_sizes) == (len(strides)) + 1 == len(sample_kernel_sizes) + 1
        self.dimensions = dimensions
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.channels = channels
        self.strides = strides
        self.kernel_sizes = kernel_sizes
        self.sample_kernel_sizes = sample_kernel_sizes
        self.num_res_units = num_res_units
        self.act = act
        self.norm = norm
        self.dropout = dropout
        self.attention_module = attention_module
        self.att_maps = []

        def _create_block(inc, outc, channels, strides, kernel_sizes, sample_kernel_sizes, is_top):
            """
            Builds the UNet2d5_spvPA structure from the bottom up by recursing down to the bottom block, then creating sequential
            blocks containing the downsample path, a skip connection around the previous block, and the upsample path.
            """
            c = channels[0]
            s = strides[0]
            k = kernel_sizes[0]
            sk = sample_kernel_sizes[0]

            # create layer in downsampling path
            down = self._get_down_layer(in_channels=inc, out_channels=c, kernel_size=k)
            downsample = self._get_downsample_layer(in_channels=c, out_channels=c, strides=s, kernel_size=sk)

            if len(channels) > 2:
                # continue recursion down
                subblock = _create_block(c, channels[1], channels[1:], strides[1:], kernel_sizes[1:],
                                         sample_kernel_sizes[1:], is_top=False)
            else:
                # the next layer is the bottom so stop recursion, create the bottom layer as the sublock for this layer
                subblock = self._get_bottom_layer(in_channels=c,
                                                  out_channels=channels[1],
                                                  kernel_size=kernel_sizes[1],
                                                  )

            upsample = self._get_upsample_layer(in_channels=channels[1], out_channels=c, strides=s, up_kernel_size=sk)
            subblock_with_resampling = nn.Sequential(downsample, subblock, upsample)

            # create layer in upsampling path
            up = self._get_up_layer(in_channels=2 * c, out_channels=outc, kernel_size=k, is_top=is_top)

            return nn.Sequential(down, SkipConnection(subblock_with_resampling), up)

        self.model = _create_block(in_channels, out_channels, self.channels, self.strides, self.kernel_sizes,
                                   self.sample_kernel_sizes, True)

        # register forward hooks on all Attentionblock1 modules, to save the attention maps
        if self.attention_module:
            for layer in self.model.modules():
                if type(layer) == AttentionBlock1:
                    layer.register_forward_hook(self.hook_save_attention_map)

    def hook_save_attention_map(self, module, inp, outp):
        if len(self.att_maps) == len(self.channels):
            self.att_maps = []
        self.att_maps.append(outp[0])  # get first element of output (Attentionblock1 returns (att, x) )

    def _get_att_layer(self, in_channels, out_channels, kernel_size):
        att1 = AttentionBlock1(
            self.dimensions,
            in_channels,
            out_channels,
            kernel_size,
            norm=None,
            dropout=self.dropout
            )

        att2 = AttentionBlock2(
            self.dimensions,
            in_channels,
            out_channels,
            kernel_size,
            norm=None,
            dropout=self.dropout
            )

        return nn.Sequential(att1, att2)

    def _get_down_layer(self, in_channels, out_channels, kernel_size):
        if self.num_res_units > 0:
            return ResidualUnit(
                self.dimensions,
                in_channels,
                out_channels,
                strides=1,
                kernel_size=kernel_size,
                subunits=self.num_res_units,
                act=self.act,
                norm=self.norm,
                dropout=self.dropout,
            )
        else:
            return Convolution(
                self.dimensions,
                in_channels,
                out_channels,
                strides=1,
                kernel_size=kernel_size,
                act=self.act,
                norm=self.norm,
                dropout=self.dropout,
            )

    def _get_downsample_layer(self, in_channels, out_channels, strides, kernel_size):
        conv = Convolution(
            self.dimensions,
            in_channels,
            out_channels,
            strides,
            kernel_size,
            self.act,
            self.norm,
            self.dropout,
            is_transposed=False,
        )
        return conv

    def _get_bottom_layer(self, in_channels, out_channels, kernel_size):
        conv = self._get_down_layer(in_channels, out_channels, kernel_size)
        if self.attention_module:
            att_layer = self._get_att_layer(in_channels, in_channels, kernel_size)
            return nn.Sequential(att_layer, conv)
        else:
            return conv


    def _get_upsample_layer(self, in_channels, out_channels, strides, up_kernel_size):
        conv = Convolution(
            self.dimensions,
            in_channels,
            out_channels,
            strides,
            up_kernel_size,
            self.act,
            self.norm,
            self.dropout,
            is_transposed=True,
        )
        return conv

    def _get_up_layer(self, in_channels, out_channels, kernel_size, is_top):

        if self.attention_module:
            att_layer = self._get_att_layer(in_channels, in_channels, kernel_size)

        if self.num_res_units > 0:
            ru = ResidualUnit(
                self.dimensions,
                in_channels,
                out_channels,
                strides=1,
                kernel_size=kernel_size,
                subunits=1,  # why not self.num_res_units?
                act=self.act,
                norm=self.norm,
                dropout=self.dropout,
                last_conv_only=is_top,
            )

        if self.attention_module and self.num_res_units > 0:
            return nn.Sequential(att_layer, ru)
        elif self.attention_module and not self.num_res_units > 0:
            return att_layer
        elif self.num_res_units > 0 and not self.attention_module:
            return ru
        elif not self.attention_module and not self.num_res_units > 0:
            return nn.Identity
        else:
            raise NotImplementedError

    def forward(self, x):
        x = self.model(x)
        return x, self.att_maps


Unet2d5_spvPA = unet2d5_spvPA = UNet2d5_spvPA
