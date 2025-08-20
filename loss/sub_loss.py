import torch
import torch.nn as nn
import torch.nn.functional as F
from util.register import LOSS_REGISTRY
from util.ssim import SSIM
from util.perception import PerceptualLoss

@LOSS_REGISTRY.register('ce')
class CrossEntropyLoss(nn.Module):
    def __init__(self, reduction = 'none', weight = 1.0):
        super().__init__()

        self.w = weight
        self.loss = nn.CrossEntropyLoss(reduction = reduction, label_smoothing = 0.02)

    def forward(self, inf, gt):
        loss = self.w * self.loss(inf, gt)
        return loss

@LOSS_REGISTRY.register('nll')
class CrossEntropyLoss(nn.Module):
    def __init__(self, reduction = 'none', weight = 1.0):
        super().__init__()

        self.w = weight
        self.loss = nn.NLLLoss(reduction = reduction)

    def forward(self, inf, gt):
        inf = torch.log(inf)
        loss = self.w * self.loss(inf, gt)
        return loss

@LOSS_REGISTRY.register('l1')
class L1Loss(nn.Module):
    def __init__(self, reduction = 'none', weight = 1.0):
        super().__init__()

        self.w = weight
        self.loss = nn.L1Loss(reduction = reduction)

    def forward(self, inf, gt):
        assert inf.shape == gt.shape, Exception('shape should be the same: inf: {}, gt: {}'.format(inf.shape, gt.shape))
        loss = self.w * self.loss(inf, gt)
        return loss

@LOSS_REGISTRY.register('smooth_l1')
class SmoothL1Loss(nn.Module):
    def __init__(self, reduction = 'none', weight = 1.0, beta = 0.02):
        super().__init__()

        self.w = weight
        self.loss = nn.SmoothL1Loss(reduction = reduction, beta = beta)

    def forward(self, inf, gt):
        assert inf.shape == gt.shape, Exception('shape should be the same: inf: {}, gt: {}'.format(inf.shape, gt.shape))
        loss = self.w * self.loss(inf, gt)
        return loss

@LOSS_REGISTRY.register('l2')
class MSELoss(nn.Module):
    def __init__(self, reduction = 'none', weight = 1.0):
        super().__init__()

        self.w = weight
        self.loss = nn.MSELoss(reduction = reduction)

    def forward(self, inf, gt):
        assert inf.shape == gt.shape, Exception('shape should be the same: inf: {}, gt: {}'.format(inf.shape, gt.shape))
        loss = self.w * self.loss(inf, gt)
        return loss

@LOSS_REGISTRY.register('ssim')
class SSIMLoss(nn.Module):
    def __init__(self, reduction = 'none', weight = 1.0):
        super().__init__()
        self.w = weight
        self.loss = SSIM()

    def forward(self, inf, gt):
        assert inf.shape == gt.shape, Exception('shape should be the same: inf: {}, gt: {}'.format(inf.shape, gt.shape))
        if len(inf.shape) == 4:
            inf = inf.permute(1,0,2,3)
            gt = gt.permute(1,0,2,3)
        loss = self.w * (1 - self.loss(inf, gt))
        return loss

@LOSS_REGISTRY.register('perception')
class PerceptionLoss(nn.Module): # TODO: need modification according to the dimension of the output & gt
    def __init__(self, reduction = 'none', weight = 1.0, device = 'cuda'):
        super().__init__()
        self.w = weight
        self.loss = PerceptualLoss(blocks = [0, 1, 2], weights = [0.3, 0.4, 0.3], device = device)

    def forward(self, inf, gt):
        assert inf.shape == gt.shape, Exception('shape should be the same: inf: {}, gt: {}'.format(inf.shape, gt.shape))
        loss = self.w * self.loss(inf, gt)
        return loss

@LOSS_REGISTRY.register('fft')
class FFTLoss(nn.Module):
    def __init__(self, reduction = 'none', weight = 1.0):
        super().__init__()
        self.w = weight
        # self.loss = nn.MSELoss(reduction = reduction)
        self.loss = nn.SmoothL1Loss(reduction = reduction, beta = 0.05)

    def forward(self, inf, gt):
        assert inf.shape == gt.shape, Exception('shape should be the same: inf: {}, gt: {}'.format(inf.shape, gt.shape))
        if len(inf.shape) == 4:
            inf = inf.permute(1,0,2,3)
            gt = gt.permute(1,0,2,3)

        inf_ft = torch.fft.fft2(inf)[...,1:, 1:]
        gt_ft = torch.fft.fft2(gt)[...,1:, 1:]
        loss = self.w * self.loss(torch.abs(inf_ft), torch.abs(gt_ft))
        return loss

@LOSS_REGISTRY.register('charbonnier')
class CharbonnierLoss(nn.Module):
    def __init__(self, reduction = 'none', weight = 1.0, eps = 1e-3):
        super().__init__()
        self.w = weight
        self.eps = eps

    def forward(self, inf, gt):
        diff = inf - gt
        loss = torch.mean(torch.sqrt(diff ** 2 + self.eps **2))
        return loss

@LOSS_REGISTRY.register('scl') # super contrast loss
class SuperContrastLoss(nn.Module):
    def __init__(self, reduction = 'none', weight = 1.0):
        super().__init__()
        self.scl_loss = SupConLoss()
    
    def forward(self, inf, gt):
        return self.scl_loss(inf, gt)

class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = features.device
        # device = (torch.device('cuda')
        #           if features.is_cuda
        #           else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss

@LOSS_REGISTRY.register('fcl')
class FocalLoss(nn.Module):
    def __init__(self, reduction = 'none', weight = 0.2, gamma = 2):
        super().__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, inf, gt, Lam):
        inf_p = F.softmax(inf, dim = 1)

        inf_p = inf_p.view(inf.size()[0], -1)
        inf_p = torch.gather(inf_p, dim = 1, index = gt.view(-1, 1))
        ce = -1 * torch.log(inf_p + 1e-7)
        floss = torch.pow((Lam - inf_p), self.gamma) * ce
        floss = torch.mul(floss, 0.2)
        floss = torch.sum(floss, dim = 1)
        return torch.mean(floss)




