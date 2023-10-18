_base_ = [
    '../_base_/models/setr_naive.py',
    '../_base_/datasets/cityscapes_768x768.py', '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_80k.py'
]

checkpoint = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segmenter/vit_small_p16_384_20220308-410f6037.pth'

norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    pretrained=checkpoint,
    backbone=dict(
        embed_dims=384,
        num_layers=12,
        num_heads=6,
        out_indices=(2, 5, 8, 11),
        final_norm=True,
        drop_rate=0.),
    decode_head=dict(
        type='SETRUPHead',
        in_channels=384),
    auxiliary_head=[
        dict(
            type='SETRUPHead',
            in_channels=384,
            channels=256,
            in_index=0,
            num_classes=19,
            dropout_ratio=0,
            norm_cfg=norm_cfg,
            num_convs=1,
            up_scale=4,
            kernel_size=1,
            align_corners=False,
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),
        dict(
            type='SETRUPHead',
            in_channels=384,
            channels=256,
            in_index=1,
            num_classes=19,
            dropout_ratio=0,
            norm_cfg=norm_cfg,
            num_convs=1,
            up_scale=4,
            kernel_size=1,
            align_corners=False,
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),
        dict(
            type='SETRUPHead',
            in_channels=384,
            channels=256,
            in_index=2,
            num_classes=19,
            dropout_ratio=0,
            norm_cfg=norm_cfg,
            num_convs=1,
            up_scale=4,
            kernel_size=1,
            align_corners=False,
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4))
    ],
    test_cfg=dict(mode='slide', crop_size=(768, 768), stride=(512, 512)),
)

optimizer = dict(
    weight_decay=0.0,
    paramwise_cfg=dict(custom_keys={'head': dict(lr_mult=10.)}))

# batch = 8
data = dict(samples_per_gpu=2, workers_per_gpu=2)
