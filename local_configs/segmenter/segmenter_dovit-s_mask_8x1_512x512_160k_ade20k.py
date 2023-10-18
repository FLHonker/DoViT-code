_base_ = [
    '../_base_/models/segmenter_vit-b16_mask.py',
    '../_base_/datasets/ade20k.py', '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_160k.py'
]

find_unused_parameters = True
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='DoEncoderDecoder',
    pretrained='pretrain/dovit_small_p16_384.pth',
    backbone=dict(
        type='DoViT',
        img_size=(512, 512),
        embed_dims=384,
        num_heads=6,
        out_indices=(2, 5, 8, 11),
        exit_conf=0.96,
        diff_conf=0.02,
        max_iters=14e4
    ),
    decode_head=dict(
        type='SegmenterMaskTransformerHead',
        in_channels=384,
        channels=384,
        num_classes=150,
        num_layers=2,
        num_heads=6,
        embed_dims=384,
        dropout_ratio=0.0,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    auxiliary_head=[
        dict(
            type='FCNHead',
            in_channels=384,
            channels=256,
            in_index=0,
            dropout_ratio=0,
            norm_cfg=norm_cfg,
            act_cfg=dict(type='ReLU'),
            num_convs=1,
            kernel_size=1,
            concat_input=False,
            num_classes=150,
            align_corners=False,
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),
        dict(
            type='FCNHead',
            in_channels=384,
            channels=256,
            in_index=0,
            dropout_ratio=0,
            norm_cfg=norm_cfg,
            act_cfg=dict(type='ReLU'),
            num_convs=1,
            kernel_size=1,
            concat_input=False,
            num_classes=150,
            align_corners=False,
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),
        dict(
            type='FCNHead',
            in_channels=384,
            channels=256,
            in_index=0,
            dropout_ratio=0,
            norm_cfg=norm_cfg,
            act_cfg=dict(type='ReLU'),
            num_convs=1,
            kernel_size=1,
            concat_input=False,
            num_classes=150,
            align_corners=False,
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),
    ],
)

optimizer = dict(lr=0.001, weight_decay=0.0)

img_norm_cfg = dict(
    mean=[127.5, 127.5, 127.5], std=[127.5, 127.5, 127.5], to_rgb=True)
crop_size = (512, 512)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=True),
    dict(type='Resize', img_scale=(2048, 512), ratio_range=(0.5, 2.0)),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(2048, 512),
        # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]
data = dict(
    # num_gpus: 8 -> batch_size: 8
    samples_per_gpu=1,
    train=dict(pipeline=train_pipeline),
    val=dict(pipeline=test_pipeline),
    test=dict(pipeline=test_pipeline)
)
