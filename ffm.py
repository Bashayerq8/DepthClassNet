
'''
Copyright (c) 2025 Bashayer Abdallah
Licensed under CC BY-NC 4.0 (https://creativecommons.org/licenses/by-nc/4.0/)
Commercial use is prohibited.

'''

# This Module is proposed in MDE-ED paper (https://ieeexplore.ieee.org/document/11084697)

import torch
from mmcv.cnn import ConvModule
import torch.nn as nn



class BaseModule(nn.Module):
    def __init__(self):
        super(BaseModule, self).__init__()



class ffm(BaseModule):
    """FeatureFusion Module for SWIN & CNN features fusion.
        Args:
            in_channels (List[int]): Number of input channels per scale.
            out_channels (int): Number of output channels.
            embedding_dim (int): Feature dimension in HAHI.
            norm_cfg (dict): Config dict for normalization layer. Default: dict(type='BN', requires_grad=True).
            act_cfg (dict): Config dict for activation layer in ConvModule. Default: dict(type='ReLU', inplace=True).
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 embedding_dim,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 act_cfg=dict(type='LeakyReLU', inplace=True)):
        super(FFM, self).__init__()
        assert isinstance(in_channels, list)
        assert len(in_channels) == 5, "Expected 5 input channels: [E, s1, s2, s3, s4]"

        self.embedding_dim = embedding_dim
        self.in_channels = in_channels
        self.out_channels = out_channels

        # 1x1 convolution to combine E and s1
        self.conv_e_s1 = ConvModule(
            in_channels[0] + in_channels[1],  # E + s1
            embedding_dim,
            kernel_size=1,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)

        # 1x1 convolution to fuse with s1
        self.fuse_s1 = ConvModule(
            embedding_dim + in_channels[1],  # conv_e_s1 + s1
            out_channels,
            kernel_size=1,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)

        # Layers for s2, s3, s4
        self.conv_proj = nn.ModuleList()
        self.fusion_layers = nn.ModuleList()
        for i in range(2, 5):
            # 1x1 convolution for initial projection
            self.conv_proj.append(ConvModule(
                in_channels[i],
                embedding_dim,
                kernel_size=1,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg))

            # 1x1 convolution for final fusion
            self.fusion_layers.append(ConvModule(
                embedding_dim + in_channels[i],
                out_channels,
                kernel_size=1,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg))

    def forward(self, x1, x2):
        # Assert that x1 contains 4 inputs: [s1, s2, s3, s4]
        assert len(x1) == 4, "Expected 4 inputs: [s1, s2, s3, s4]"

        # Assert that x2 contains 1 input, which is E
        assert len(x2) == 1, "Expected 1 input for x2, which is [E]"

        # Unpack x1 and x2 correctly
        s1, s2, s3, s4 = x1
        E = x2[0]

        # Step 1: Concatenate E and s1, then apply 1x1 convolution
        e_s1_cat = torch.cat([E, s1], dim=1)  # Concatenate along channel dimension
        e_s1_fused = self.conv_e_s1(e_s1_cat)

        # Step 2: Concatenate the result with s1, then apply another 1x1 convolution to produce f1
        s1_fused = torch.cat([e_s1_fused, s1], dim=1)
        f1 = self.fuse_s1(s1_fused)

        # Step 3: Process s2, s3, s4
        f2_to_f4 = []
        for i, s in enumerate([s2, s3, s4]):
            # Apply initial 1x1 convolution
            proj_feat = self.conv_proj[i](s)

            # Concatenate with original feature map (s2, s3, or s4)
            fused_feat = torch.cat([proj_feat, s], dim=1)

            # Apply final 1x1 convolution to produce the output
            fused_output = self.fusion_layers[i](fused_feat)
            f2_to_f4.append(fused_output)

        # Return the final outputs as a tuple
        return (f1, *f2_to_f4)


