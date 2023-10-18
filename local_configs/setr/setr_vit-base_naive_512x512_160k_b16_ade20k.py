_base_ = [
    '../_base_/models/setr_naive.py', '../_base_/datasets/ade20k.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_160k.py'
]

checkpoint = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segmenter/vit_base_p16_384_20220308-96dfe169.pth'  # noqa

norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    pretrained=checkpoint,
    backbone=dict(
        img_size=(512, 512),
        embed_dims=768,
        num_layers=12,
        num_heads=12,
        out_indices=(2, 5, 8, 11),
        # final_norm=True,
        drop_rate=0.),
    decode_head=dict(
        type='SETRUPHead',
        in_channels=768,
        in_index=-1,
        num_classes=150),
    auxiliary_head=[
        dict(
            type='SETRUPHead',
            in_channels=768,
            channels=256,
            in_index=0,
            num_classes=150,
            dropout_ratio=0,
            norm_cfg=norm_cfg,
            act_cfg=dict(type='ReLU'),
            num_convs=1,
            up_scale=4,
            kernel_size=1,
            align_corners=False,
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),
        dict(
            type='SETRUPHead',
            in_channels=768,
            channels=256,
            in_index=1,
            num_classes=150,
            dropout_ratio=0,
            norm_cfg=norm_cfg,
            act_cfg=dict(type='ReLU'),
            num_convs=1,
            up_scale=4,
            kernel_size=1,
            align_corners=False,
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),
        dict(
            type='SETRUPHead',
            in_channels=768,
            channels=256,
            in_index=2,
            num_classes=150,
            dropout_ratio=0,
            norm_cfg=norm_cfg,
            act_cfg=dict(type='ReLU'),
            num_convs=1,
            up_scale=4,
            kernel_size=1,
            align_corners=False,
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4))
    ],
    test_cfg=dict(mode='slide', crop_size=(512, 512), stride=(341, 341)),
)

optimizer = dict(
    weight_decay=0.0,
    paramwise_cfg=dict(custom_keys={'head': dict(lr_mult=10.)}))

# batch = 16
data = dict(samples_per_gpu=2, workers_per_gpu=2)
