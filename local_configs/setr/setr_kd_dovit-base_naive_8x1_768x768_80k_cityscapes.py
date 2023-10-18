_base_ = [
    './setr_dovit-base_naive_8x1_768x768_80k_cityscapes.py'
]

find_unused_parameters=True
distiller = dict(
    type='DoViTDistiller',
    teacher_pretrained = 'work_dirs/setr_vit-base_naive_8x1_768x768_80k_cityscapes/iter_80000.pth',
    distill_cfg = dict(tau=1.0, kl_weight=1.0, kd_weight=0.5)
    )

student_cfg = 'kd_configs/setr/setr_dovit-base_naive_8x1_768x768_80k_cityscapes.py'
teacher_cfg = 'kd_configs/setr/setr_vit-base_naive_8x1_768x768_80k_cityscapes.py'

data = dict(samples_per_gpu=1, workers_per_gpu=1)