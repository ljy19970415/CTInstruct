
from typing import Sequence, Tuple, Union

import torch
import torch.nn as nn

from monai.networks.blocks.dynunet_block import UnetBasicBlock, UnetResBlock, get_conv_layer
from einops import rearrange

class UnetrDecoder(nn.Module):
    def __init__(self) -> None:
        super(UnetrDecoder, self).__init__()
        norm_name = "instance"
        norm_name = "batch"
        self.decoder_vit = UnetrUpBlock(
            spatial_dims=2,
            in_channels=2048,
            out_channels=2048,
            stride=1,
            kernel_size=3,
            upsample_kernel_size=8,
            norm_name=norm_name,
            res_block=False,
        )

        self.decoder1 = UnetrUpBlock(
            spatial_dims=2,
            in_channels=2048,
            out_channels=1024,
            stride=1,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=False,
        )

        self.decoder2 = UnetrUpBlock(
            spatial_dims=2,
            in_channels=1024,
            out_channels=512,
            stride=1,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=False,
        )

        self.decoder3 = UnetrUpBlock(
            spatial_dims=2,
            in_channels=512,
            out_channels=256,
            stride=1,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=False,
        )

        self.decoder4 = UnetrUpBlock(
            spatial_dims=2,
            in_channels=256,
            out_channels=64,
            stride=1,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=False,
        )
        self.transp_conv = get_conv_layer(
            spatial_dims=2,
            in_channels=64,
            out_channels=64,
            kernel_size=2,
            stride=2,
            conv_only=True,
            is_transposed=True,
        )

    def get_last_layer(self,vit_patch, skips, idx):
        patch1 = self.decoder_vit(vit_patch[idx], skips[-1][idx]) # 32 2048 8 8
        dec1 = self.decoder1(patch1,skips[-2][idx]) # 32 1024 16 16
        del patch1
        dec2 = self.decoder2(dec1,skips[-3][idx]) # 32 512 32 32
        del dec1
        dec3 = self.decoder3(dec2,skips[-4][idx]) # 32 256 64 64
        del dec2
        dec4 = self.decoder4(dec3,skips[-5][idx]) # 32 64 128 128
        del dec3
        dec5 = self.transp_conv(dec4) # 32 64 256 256
        return dec5

    def forward(self,skips, vit_patch, split_size=None):

        vit_patch = vit_patch.unsqueeze(-1).unsqueeze(-1)
        vit_patch = rearrange(vit_patch, 'b d c h w -> (b d) c h w')
        
        if split_size is not None:
            last_layer_feature = []
            vit_patch = torch.split(vit_patch, split_size, dim=0) # [(z_real d h w), (z_real d h w),..N]

            skips = list(map(lambda x: torch.split(x, split_size, dim=0), skips))

            for idx in range(len(vit_patch)):
                dec5 = self.get_last_layer(vit_patch,skips,idx)
                torch.cuda.empty_cache()
            return torch.cat(last_layer_feature, dim=0)
        else:
            vit_patch = self.decoder_vit(vit_patch,skips[-1])
            dec1 = self.decoder1(vit_patch,skips[-2]) # 32 1024 16 16
            del vit_patch
            dec2 = self.decoder2(dec1,skips[-3]) # 32 512 32 32
            del dec1
            dec3 = self.decoder3(dec2,skips[-4]) # 32 256 64 64
            del dec2
            dec4 = self.decoder4(dec3,skips[-5]) # 32 64 128 128
            del dec3
            dec5 = self.transp_conv(dec4) # 32 64 256 256
            del dec4
            return dec5
        

class UnetrUpBlock(nn.Module):
    """
    An upsampling module that can be used for UNETR: "Hatamizadeh et al.,
    UNETR: Transformers for 3D Medical Image Segmentation <https://arxiv.org/abs/2103.10504>"
    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,  # type: ignore
        kernel_size: Union[Sequence[int], int],
        stride: Union[Sequence[int], int],
        upsample_kernel_size: Union[Sequence[int], int],
        norm_name: Union[Tuple, str],
        res_block: bool = False,
    ) -> None:
        """
        Args:
            spatial_dims: number of spatial dimensions.
            in_channels: number of input channels.
            out_channels: number of output channels.
            kernel_size: convolution kernel size.
            stride: convolution stride.
            upsample_kernel_size: convolution kernel size for transposed convolution layers.
            norm_name: feature normalization type and arguments.
            res_block: bool argument to determine if residual block is used.

        """

        super(UnetrUpBlock, self).__init__()
        upsample_stride = upsample_kernel_size
        self.transp_conv = get_conv_layer(
            spatial_dims,
            in_channels,
            out_channels,
            kernel_size=upsample_kernel_size,
            stride=upsample_stride,
            conv_only=True,
            is_transposed=True,
        )

        if res_block:
            self.conv_block = UnetResBlock(
                spatial_dims,
                out_channels + out_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=1,
                norm_name=norm_name,
            )
        else:
            self.conv_block = UnetBasicBlock(  # type: ignore
                spatial_dims,
                out_channels + out_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=1,
                norm_name=norm_name,
            )

    def forward(self, inp, skip):
        # number of channels for skip should equals to out_channels
        out = self.transp_conv(inp)
        out = torch.cat((out, skip), dim=1)
        out = self.conv_block(out)
        return out



class UnetrPrUpBlock(nn.Module):
    """
    A projection upsampling module that can be used for UNETR: "Hatamizadeh et al.,
    UNETR: Transformers for 3D Medical Image Segmentation <https://arxiv.org/abs/2103.10504>"
    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        num_layer: int,
        kernel_size: Union[Sequence[int], int],
        stride: Union[Sequence[int], int],
        upsample_kernel_size: Union[Sequence[int], int],
        norm_name: Union[Tuple, str],
        conv_block: bool = False,
        res_block: bool = False,
    ) -> None:
        """
        Args:
            spatial_dims: number of spatial dimensions.
            in_channels: number of input channels.
            out_channels: number of output channels.
            num_layer: number of upsampling blocks.
            kernel_size: convolution kernel size.
            stride: convolution stride.
            upsample_kernel_size: convolution kernel size for transposed convolution layers.
            norm_name: feature normalization type and arguments.
            conv_block: bool argument to determine if convolutional block is used.
            res_block: bool argument to determine if residual block is used.

        """

        super().__init__()

        upsample_stride = upsample_kernel_size
        self.transp_conv_init = get_conv_layer(
            spatial_dims,
            in_channels,
            out_channels,
            kernel_size=upsample_kernel_size,
            stride=upsample_stride,
            conv_only=True,
            is_transposed=True,
        )
        if conv_block:
            if res_block:
                self.blocks = nn.ModuleList(
                    [
                        nn.Sequential(
                            get_conv_layer(
                                spatial_dims,
                                out_channels,
                                out_channels,
                                kernel_size=upsample_kernel_size,
                                stride=upsample_stride,
                                conv_only=True,
                                is_transposed=True,
                            ),
                            UnetResBlock(
                                spatial_dims=3,
                                in_channels=out_channels,
                                out_channels=out_channels,
                                kernel_size=kernel_size,
                                stride=stride,
                                norm_name=norm_name,
                            ),
                        )
                        for i in range(num_layer)
                    ]
                )
            else:
                self.blocks = nn.ModuleList(
                    [
                        nn.Sequential(
                            get_conv_layer(
                                spatial_dims,
                                out_channels,
                                out_channels,
                                kernel_size=upsample_kernel_size,
                                stride=upsample_stride,
                                conv_only=True,
                                is_transposed=True,
                            ),
                            UnetBasicBlock(
                                spatial_dims=3,
                                in_channels=out_channels,
                                out_channels=out_channels,
                                kernel_size=kernel_size,
                                stride=stride,
                                norm_name=norm_name,
                            ),
                        )
                        for i in range(num_layer)
                    ]
                )
        else:
            self.blocks = nn.ModuleList(
                [
                    get_conv_layer(
                        spatial_dims,
                        out_channels,
                        out_channels,
                        kernel_size=upsample_kernel_size,
                        stride=upsample_stride,
                        conv_only=True,
                        is_transposed=True,
                    )
                    for i in range(num_layer)
                ]
            )

    def forward(self, x):
        x = self.transp_conv_init(x)
        for blk in self.blocks:
            x = blk(x)
        return x


class UnetrBasicBlock(nn.Module):
    """
    A CNN module that can be used for UNETR, based on: "Hatamizadeh et al.,
    UNETR: Transformers for 3D Medical Image Segmentation <https://arxiv.org/abs/2103.10504>"
    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[Sequence[int], int],
        stride: Union[Sequence[int], int],
        norm_name: Union[Tuple, str],
        res_block: bool = False,
    ) -> None:
        """
        Args:
            spatial_dims: number of spatial dimensions.
            in_channels: number of input channels.
            out_channels: number of output channels.
            kernel_size: convolution kernel size.
            stride: convolution stride.
            norm_name: feature normalization type and arguments.
            res_block: bool argument to determine if residual block is used.

        """

        super().__init__()

        if res_block:
            self.layer = UnetResBlock(
                spatial_dims=spatial_dims,
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                norm_name=norm_name,
            )
        else:
            self.layer = UnetBasicBlock(  # type: ignore
                spatial_dims=spatial_dims,
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                norm_name=norm_name,
            )

    def forward(self, inp):
        out = self.layer(inp)
        return out
