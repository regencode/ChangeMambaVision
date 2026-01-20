from timm.models.registry import register_model
import torch
from torch import  nn
import torch.nn.functional as F
from einops import *
from .mamba_vision import MambaVision, MambaVisionMixer
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
import math
from .MambaVisionModels import create_model, register_pip_model, list_models
from .ChangeFormerModels import DecoderTransformer_v3

class MambaVisionCD(nn.Module):
    def __init__(self,
                 in_chans,
                 decoder_model="changeformer",
                 encoder_model=None,
                 dims=[64, 128, 256, 512],
                 embed_dims=256,
                 reduced_dims=None,
                 depths=[2, 2, 4, 2],
                 window_size=[4, 4, 6, 8],
                 mlp_ratio=4,
                 num_heads=[2, 4, 8, 16],
                 drop_path_rate=0.1,
                 num_classes=2,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.1,
                 attn_drop_rate=0.1,
                 layer_scale=None,
                 layer_scale_conv=None,
                 patchembed_downsample=True,
                 **kwargs):
        super().__init__()
        if encoder_model is not None:
            # use checkpoints
            self.enc = create_model(encoder_model, in_chans=in_chans, patchembed_downsample=patchembed_downsample, **kwargs)
        else:
            self.enc = MambaVision(
                     in_chans,
                     dims,
                     depths,
                     window_size,
                     mlp_ratio,
                     num_heads,
                     drop_path_rate=drop_path_rate,
                     num_classes=num_classes,
                     qkv_bias=qkv_bias,
                     qk_scale=qk_scale,
                     drop_rate=drop_rate,
                     attn_drop_rate=attn_drop_rate,
                     layer_scale=layer_scale,
                     layer_scale_conv=layer_scale_conv,
                     patchembed_downsample=patchembed_downsample
            )
        self.dec = nn.Identity()
        self.decoder_model = decoder_model
        print(f"using downsample={patchembed_downsample}")
        print(f"using dims={self.enc.dims}")
        self.decoder = DecoderTransformer_v3(input_transform='multiple_select', in_index=[0, 1, 2, 3], align_corners=False, 
                                                in_channels=self.enc.dims, embedding_dim=embed_dims, output_nc=num_classes, 
                                                decoder_softmax=False, feature_strides=[2, 4, 8, 16], final_upsample=[False, False])
    def forward(self, x1, x2):
        _, _, H, W = x1.shape
        x1s = self.enc(x1)
        x2s = self.enc(x2)
        return self.decoder(x1s, x2s)[-1]

if __name__ == "__main__":
    print(list_models())
