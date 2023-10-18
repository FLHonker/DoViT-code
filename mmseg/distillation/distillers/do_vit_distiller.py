import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

from mmseg.core import add_prefix
from mmseg.ops import resize
from mmcv.runner import load_checkpoint
from ..builder import DISTILLER, build_distill_loss
from mmseg.models import build_segmentor
from mmseg.models.segmentors.base import BaseSegmentor

from ..losses import PixelKL, CWD, SKD, AT, MGD

@DISTILLER.register_module()
class DoViTDistiller(BaseSegmentor):
    """Dynamic output distiller for segmentors.

    It typically consists of one dynamic-output student_model and one pre-trained full-infer student_model.
    """
    def __init__(self,
                 teacher_cfg,
                 student_cfg,
                 distill_cfg=None,
                 teacher_pretrained=None,):

        super(DoViTDistiller, self).__init__()
        
        self.teacher = build_segmentor(teacher_cfg.model,
                                        train_cfg=teacher_cfg.get('train_cfg'),
                                        test_cfg=teacher_cfg.get('test_cfg'))
        self.init_weights_teacher(teacher_pretrained)
        self.teacher.eval()
        
        self.student = build_segmentor(student_cfg.model,
                                        train_cfg=student_cfg.get('train_cfg'),
                                        test_cfg=student_cfg.get('test_cfg'))
        self.student.init_weights()
        
        self.distill_cfg = distill_cfg
        self.distill_losses = nn.ModuleDict()
        self.distill_losses['logits'] = PixelKL(tau=distill_cfg.tau, weight=distill_cfg.kl_weight)
        self.distill_losses['feats'] = nn.MSELoss()

        teacher_channels = 768
        self.generation = nn.Sequential(
            nn.Conv2d(teacher_channels, teacher_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True), 
            nn.Conv2d(teacher_channels, teacher_channels, kernel_size=1)
        )
    
    def base_parameters(self):
        return nn.ModuleList([self.student, self.distill_losses, self.generation])

    def init_weights_teacher(self, path=None):
        """Load the pretrained model in teacher segmentor.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """
        checkpoint = load_checkpoint(self.teacher, path, map_location='cpu')
        # checkpoint = load_checkpoint(self.student, path, map_location='cpu')


    def forward_train(self, img, img_metas, gt_semantic_seg):
        """Forward function for training.

        Args:
            img (Tensor): Input images.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            gt_semantic_seg (Tensor): Semantic segmentation masks
                used if the architecture supports semantic segmentation task.

        Returns:
            dict[str, Tensor]: a dictionary of loss components. 
        """

        with torch.no_grad():
            self.teacher.eval()
            feats_t = self.teacher.extract_feat(img)
            logit_t = self.teacher._decode_head_forward_test(feats_t, img_metas)

        feats_s, aux_logits = self.student.extract_feat(img)
        logit_s = self.student._decode_head_forward_test(feats_s, img_metas)
        student_loss = self.student.decode_head.losses(logit_s, gt_semantic_seg)

        loss_aux = self.student._auxiliary_head_forward_train(
            aux_logits, img_metas, gt_semantic_seg)
        student_loss.update(loss_aux)
        
        f_t, f_s = feats_t[-1], feats_s[-1]
        # f_t = F.normalize(f_t, dim=1)
        # f_s = F.normalize(f_s, dim=1)
        # for idx, (f_s, f_t) in enumerate(zip(feats_s[:3], feats_t[:3])):
        #     mask = (torch.sum(f_s, dim=1, keepdim=True) != 0).float().detach()
        #     f_t = f_t * mask
        #     f_s = f_s * mask
        #     student_loss[f'loss_KD_{idx}'] = self.distill_cfg.kd_weight * self.distill_losses['feats'](f_s, f_t.detach())

        student_loss['loss_KD'] = self.distill_cfg.kd_weight * self.distill_losses['feats'](f_s, f_t.detach())
        student_loss['loss_KL'] = self.distill_losses['logits'](logit_s, logit_t.detach())

        return student_loss

    
    def state_dict(self):
        """ Rewrite `state_dict()` of Distiller models.
        """
        return self.student.state_dict()

    def inference(self, img, img_meta, rescale):
        return self.student.inference(img, img_meta, rescale)

    def simple_test(self, img, img_metas, rescale=True):
        return self.student.simple_test(img, img_metas, rescale)

    def aug_test(self, imgs, img_metas, rescale=True):
        return self.student.aug_test(imgs, img_metas, rescale)

    def extract_feat(self, imgs):
        return self.student.extract_feat(imgs)

    def encode_decode(self, img, img_metas):
        return self.student.encode_decode(img, img_metas)