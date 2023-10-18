import torch
import torch.nn as nn
import torch.nn.functional as F
from ..builder import DISTILL_LOSSES


@DISTILL_LOSSES.register_module()
class PixelKL(nn.Module):
    """ Paper: Distilling the Knowledge in a Neural Network, 2015.
     <https://arxiv.org/abs/1503.02531> _.
    """

    def __init__(self, tau=1.0, weight=1.0):
        super(PixelKL, self).__init__()
        self.tau = tau
        self.loss_weight = weight

    def forward(self, logits_s, logits_t):
        logits_t = logits_t.detach()
        B, C, h, w = logits_t.size()
        scale_pred = logits_s.permute(0,2,3,1).contiguous().view(-1, C)
        scale_soft = logits_t.permute(0,2,3,1).contiguous().view(-1, C)
        p_s = F.log_softmax(scale_pred / self.tau, dim=1)
        p_t = F.softmax(scale_soft / self.tau, dim=1)
        loss = F.kl_div(p_s, p_t, reduction='batchmean') * (self.tau**2)
        return self.loss_weight * loss

        # logits_t = logits_t.detach()
        # p_s = F.log_softmax(logits_s/self.tau, dim=1)
        # p_t = F.softmax(logits_t/self.tau, dim=1)
        # loss = F.kl_div(p_s, p_t, reduction='mean') * (self.tau ** 2)
        # return self.loss_weight * loss 
