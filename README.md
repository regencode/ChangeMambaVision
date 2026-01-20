# ChangeMambaVision: Adapting MambaVision for Change Detection

![ChangeMambaVision](images/ChangeMambaVision.svg)


## Metrics

![Metrics](images/metrics.png)


## Changes from original -> custom MambaVision

- in_dims -> patch_embed_dim (default to 256)
- dims in list form for each level instead of single integer for the first level
- resolution 224 -> 256 to make it in line with CD patches
- MambaVision forward() will return outputs at multiple levels

## Licenses

Copyright © 2025, NVIDIA Corporation. All rights reserved.

This work is made available under the NVIDIA Source Code License-NC. Click [here](LICENSE) to view a copy of this license.

The pre-trained models are shared under [CC-BY-NC-SA-4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/). If you remix, transform, or build upon the material, you must distribute your contributions under the same license as the original.

For license information regarding the timm repository, please refer to its [repository](https://github.com/rwightman/pytorch-image-models).

For license information regarding the ImageNet dataset, please see the [ImageNet official website](https://www.image-net.org/). 

