_base_ = [
    '../_base_/models/segmenter_vit-b16_mask.py',
    '../_base_/datasets/cityscapes_768x768.py', '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_80k.py'
]

model = dict(
    backbone=dict(img_size=(768, 768)),
    decode_head=dict(num_classes=19),
    test_cfg=dict(mode='slide', crop_size=(768, 768), stride=(512, 512)))

optimizer = dict(lr=0.01, weight_decay=0.0)

data = dict(samples_per_gpu=2, workers_per_gpu=2)
