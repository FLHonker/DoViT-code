# DoViT-code

PyTorch implementation of our paper "[Dynamic Token-Pass Transformers for Semantic Segmentation](https://arxiv.org/abs/2308.01944)" to appear in WACV'24.


## Installation

Requirements:
```
python >= 3.6
mmsegmentation >= 0.25.0
```
Please see the [documentation](https://github.com/open-mmlab/mmsegmentation/blob/main/docs/en/get_started.md#installation) of mmsegmentation for details.

## Get Started

1. Download mmseg pretrained backbones (links in the config files), and convert to DoViT architecture by:
   
```bash
bash tools/model_converters/mmseg2dovit.py </path/to/pretrained_backbone> </path/to/output_backbone>
```
Note: We modified the `EncoderDecoder` class in mmseg to support DoViT backbones.

2. Training on 8 GPUs:

```bash
bash tools/dist_train.sh </path/config_file.py> 8 --work-dir <output_dir>
```

## Citation

```
@inproceedings{liu2024dovit,
    author    = {Liu, Yuang and Zhou, Qiang and Wang, Jin and Wang, Zhibin and Wang, Fan and Wang, Jun and Zhang, Wei},
    title     = {Dynamic Token-Pass Transformers for Semantic Segmentation},
    booktitle = {IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
    month     = {January},
    year      = {2024}
}
```