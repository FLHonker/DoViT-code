_base_ = [
    '../_base_/models/segmenter_vit-b16_mask.py',
    '../_base_/datasets/cityscapes_768x768.py', '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_80k.py'
]

checkpoint = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segmenter/vit_small_p16_384_20220308-410f6037.pth'
model = dict(
    pretrained=checkpoint,
    backbone=dict(
        img_size=(768, 768),
        embed_dims=384,
        num_heads=6),
    decode_head=dict(
        in_channels=384,
        channels=384,
        num_classes=19,
        num_heads=6,
        embed_dims=384),
    test_cfg=dict(mode='slide', crop_size=(768, 768), stride=(512, 512)))

optimizer = dict(lr=0.01, weight_decay=0.0)

data = dict(samples_per_gpu=2, workers_per_gpu=2)
