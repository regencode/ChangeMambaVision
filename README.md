<h1 align="center" style="font-size:48px; font-weight:600;">
  ChangeMambaVision: 
  Adapting MambaVision for Building Change Detection
</h1>

![ChangeMambaVision](images/ChangeMambaVision.svg)


## Metrics

![Metrics](images/metrics.png)


## Changes from original -> custom MambaVision

- in_dims -> patch_embed_dim (default to 256)
- dims in list form for each level instead of single integer for the first level
- resolution 224 -> 256 to make it in line with CD patches
- MambaVision forward() will return outputs at multiple levels

## Available checkpoints

ChangeMambaVision should work with any checkpoint of MambaVision. However we only tested the -T, -S, and -B checkpoints.

Please raise a github issue if there are 
some incompatibility of a checkpoint with ChangeFormer due to 
e.g. mismatched encoder-decoder channels or import error.

## Datasets

The LEVIR-CD dataset can be downloaded from [the official website](https://justchenhao.github.io/LEVIR/).

The WHU-CD dataset can be downloaded from [the official website](https://gpcv.whu.edu.cn/data/building_dataset.html) at Section 4 - Building change detection dataset.


## Licenses

Copyright © 2025, NVIDIA Corporation. All rights reserved.

This work is made available under the NVIDIA Source Code License-NC. Click [here](LICENSE) to view a copy of this license.

The pre-trained models are shared under [CC-BY-NC-SA-4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/). If you remix, transform, or build upon the material, you must distribute your contributions under the same license as the original.

For license information regarding the timm repository, please refer to its [repository](https://github.com/rwightman/pytorch-image-models).

For license information regarding the ImageNet dataset, please see the [ImageNet official website](https://www.image-net.org/). 

