_base_ = [
    '../_base_/models/segmenter_vit-b16_mask.py',
    '../_base_/datasets/cityscapes_768x768.py', '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_80k.py'
]

find_unused_parameters = True
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='DoEncoderDecoder',
    pretrained='pretrain/dovit_small_p16_384.pth',
    backbone=dict(
        type='DoViT',
        img_size=(768, 768),
        embed_dims=384,
        num_heads=6,
        out_indices=(2, 5, 8, 11),
        exit_conf=0.985,
        diff_conf=0.0,
        max_iters=7e4,
    ),
    decode_head=dict(
        in_channels=384,
        channels=384,
        num_classes=19,
        num_heads=6,
        embed_dims=384,
        in_index=-1
    ),
    # auxiliary_head=[
    #     dict(
    #         type='SETRUPHead',
    #         in_channels=384,
    #         channels=256,
    #         in_index=0,
    #         num_classes=19,
    #         dropout_ratio=0,
    #         norm_cfg=norm_cfg,
    #         num_convs=1,
    #         up_scale=4,
    #         kernel_size=1,
    #         align_corners=False,
    #         loss_decode=dict(
    #             type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),
    #     dict(
    #         type='SETRUPHead',
    #         in_channels=384,
    #         channels=256,
    #         in_index=-1,
    #         num_classes=19,
    #         dropout_ratio=0,
    #         norm_cfg=norm_cfg,
    #         num_convs=1,
    #         up_scale=4,
    #         kernel_size=1,
    #         align_corners=False,
    #         loss_decode=dict(
    #             type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),
    #     dict(
    #         type='SETRUPHead',
    #         in_channels=384,
    #         channels=256,
    #         in_index=-1,
    #         num_classes=19,
    #         dropout_ratio=0,
    #         norm_cfg=norm_cfg,
    #         num_convs=1,
    #         up_scale=4,
    #         kernel_size=1,
    #         align_corners=False,
    #         loss_decode=dict(
    #             type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4))
    # ],
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
            num_classes=19,
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
            num_classes=19,
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
            num_classes=19,
            align_corners=False,
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),
    ],
    test_cfg=dict(mode='slide', crop_size=(768, 768), stride=(512, 512)))

optimizer = dict(lr=0.01, weight_decay=0.0)

data = dict(samples_per_gpu=1, workers_per_gpu=1)