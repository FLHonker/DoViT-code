# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os.path as osp
from collections import OrderedDict

import mmcv
import torch
from mmcv.runner import CheckpointLoader


def convert_vit(ckpt):

    new_ckpt = OrderedDict()

    for k, v in ckpt.items():
        if k.startswith('layers'):
            if 'attn.attn.in_proj_' in k:
                new_k = k.replace('attn.attn.in_proj_', 'attn.qkv.')
            elif 'attn.attn.out_proj' in k:
                new_k = k.replace('attn.attn.out_proj', 'attn.proj')
            else:
                new_k = k
        else:
            new_k = k
        new_ckpt[new_k] = v

    return new_ckpt


def main():
    parser = argparse.ArgumentParser(
        description='Convert keys in timm pretrained vit models to '
        'MMSegmentation style.')
    parser.add_argument('src', help='src model path or url')
    # The dst path must be a full path of the new checkpoint.
    parser.add_argument('dst', help='save path')
    args = parser.parse_args()

    checkpoint = CheckpointLoader.load_checkpoint(args.src, map_location='cpu')
    if 'state_dict' in checkpoint:
        # timm checkpoint
        state_dict = checkpoint['state_dict']
    elif 'model' in checkpoint:
        # deit checkpoint
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint
    weight = convert_vit(state_dict)
    mmcv.mkdir_or_exist(osp.dirname(args.dst))
    torch.save(weight, args.dst)


if __name__ == '__main__':
    main()
