import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange
import torch.distributed as dist

class BinaryDice_with_BCE_Loss(nn.Module):
    def __init__(self,):
        super(BinaryDice_with_BCE_Loss, self).__init__()
        self.dice_criterion=BinaryDiceLoss(reduction='mean')
        self.bce_criterion=nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, logits, target):
        """
        predict: A tensor of shape [B, C, H, W, D], value 0~1
        target: A binary tensor of shape same with predict
        """

        torch.cuda.empty_cache()

        mask = target != -1
        target[~mask] = 0
        bce_loss = self.bce_criterion(logits, target)
        bce_loss[~mask] = 0
        bce_loss = bce_loss.sum() / (mask.sum() + 1e-6)
        dice_loss=self.dice_criterion(logits, target, mask)
        return bce_loss,dice_loss

class BinaryDiceLoss(nn.Module):
    """
    Dice loss of binary class
    
    Args:
        smooth: A float number to smooth loss, and avoid NaN error, default: 1
        p: Denominator value: \sum{x^p} + \sum{y^p}, default: 2
        reduction: Reduction method to apply, return mean over batch if 'mean',
            return sum if 'sum', return a tensor of shape [N,] if 'none'
    Returns:
        Loss tensor according to arg reduction
    Raise:
        Exception if unexpected reduction
    """
    def __init__(self, smooth=1e-7, p=2, reduction='mean'):
        super(BinaryDiceLoss, self).__init__()
        self.smooth = smooth
        self.p = p
        self.reduction = reduction

    def forward(self, logits, target, mask):
        """
        predict: A tensor of shape [B, C, H, W, D], value 0~1
        target: A binary tensor of shape same with predict
        """
        dic = {}
        predict = torch.sigmoid(logits)
        predict = predict * mask
        assert predict.shape == target.shape, f'predict {predict.shape} & target {target.shape} do not match'
        predict = rearrange(predict.contiguous(), 'b c h w d -> (b c) (h w d)')   # B*C, H*W*D 
        target = rearrange(target.contiguous(), 'b c h w d -> (b c) (h w d)')
        intersection = 2 * torch.sum(torch.mul(predict, target), dim=1) + self.smooth
        
        predict_p = predict.pow(self.p)
        predict_p.add_(target.pow(self.p))
        union = torch.sum(predict_p, dim=1) + self.smooth
        del predict_p

        loss = 1 - intersection / union

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss
        else:
            raise Exception('Unexpected reduction {}'.format(self.reduction))
        
class DINOLoss(nn.Module):
    def __init__(self, out_dim, ncrops, warmup_teacher_temp, teacher_temp,
                 warmup_teacher_temp_iters, niters, student_temp=0.1,
                 center_momentum=0.9):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.ncrops = ncrops
        self.register_buffer("center", torch.zeros(1, out_dim))
        # we apply a warm up for the teacher temperature because
        # a too high temperature makes the training instable at the beginning
        self.teacher_temp_schedule = np.concatenate((
            np.linspace(warmup_teacher_temp,
                        teacher_temp, warmup_teacher_temp_iters),
            np.ones(niters - warmup_teacher_temp_iters) * teacher_temp
        ))

    def forward(self, student_output, teacher_output, iters):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """
        student_out = student_output / self.student_temp
        student_out = student_out.chunk(self.ncrops)

        # teacher centering and sharpening
        temp = self.teacher_temp_schedule[iters]
        teacher_out = F.softmax((teacher_output - self.center) / temp, dim=-1) #  2,65536
        teacher_out = teacher_out.detach().chunk(2) # chunks[0] = (2/2, 65536) chunks[1] = (2/2, 65536)

        total_loss = 0
        n_loss_terms = 0
        for iq, q in enumerate(teacher_out):
            for v in range(len(student_out)):
                loss = torch.sum(-q * F.log_softmax(student_out[v], dim=-1), dim=-1)
                total_loss += loss.mean()
                n_loss_terms += 1
        total_loss /= n_loss_terms
        self.update_center(teacher_output)
        return total_loss

    @torch.no_grad()
    def update_center(self, teacher_output):
        """
        Update center used for teacher output.
        """
        batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
        dist.all_reduce(batch_center)
        batch_center = batch_center / (len(teacher_output) * dist.get_world_size())

        # ema update
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)

class SampleWiseBinaryDiceLoss(nn.Module):
    """
    sample-wise Dice Loss
    """
    def __init__(self, smooth=1e-7, p=2, reduction='mean'):
        super(SampleWiseBinaryDiceLoss, self).__init__()
        self.smooth = smooth
        self.p = p
        self.reduction = reduction

    def forward(self, predict, target):
        """
        predict: A tensor of shape [C, H, W, D], value 0~1
        target: A binary tensor of shape same with predict
        """
        assert predict.shape == target.shape, f'predict {predict.shape} & target {target.shape} do not match'
        
        predict = rearrange(predict.contiguous(), 'c h w d -> c (h w d)')   # B*C, H*W*D 
        target = rearrange(target.contiguous(), 'c h w d -> c (h w d)')

        intersection = 2 * torch.sum(torch.mul(predict, target), dim=1) + self.smooth
        union = torch.sum(predict.pow(self.p) + target.pow(self.p), dim=1) + self.smooth

        loss = 1 - intersection / union

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss
        else:
            raise Exception('Unexpected reduction {}'.format(self.reduction))

def segmentation_loss(logits, mask, query_mask, dice_loss, bce_w_logits_loss, weight):
    """
    Calculate Weighted Dice and BCE Loss

    Args:
        logits (tensor): unsigmoided prediction, bnhwd
        mask (tensor): binary, bnhwd
        query_mask (tensor): binary, n
        dice_loss (_type_): loss calculator
        bce_w_logits_loss (_type_): loss calculator
        weight (float): _description_
    """
    prediction = torch.sigmoid(logits) 
    batch_dice_loss = dice_loss(prediction, mask)   # (b*n)
    batch_dice_loss = rearrange(batch_dice_loss, '(b c) -> b c', b=prediction.shape[0]) # b n
    batch_dice_loss = batch_dice_loss * query_mask
    reduced_batch_dice_loss = torch.sum(batch_dice_loss) / (torch.sum(query_mask) + 1e-14)  # bn -> 1 # NOTE: avg over all sample-label
    unreduced_batch_dice_loss = torch.sum(batch_dice_loss, dim=1) / (torch.sum(query_mask, dim=1) + 1e-14).detach()  # bn -> b

    batch_ce_loss = bce_w_logits_loss(logits, mask)  # (b, n, h, w, d)
    batch_ce_loss = torch.mean(batch_ce_loss, dim=(2,3,4)) # b n
    batch_ce_loss = batch_ce_loss * query_mask
    reduced_batch_ce_loss = torch.sum(batch_ce_loss) / (torch.sum(query_mask) + 1e-14)  # bn -> 1 # NOTE: avg over all sample-label
    unreduced_batch_ce_loss = torch.sum(batch_ce_loss, dim=1) / (torch.sum(query_mask, dim=1) + 1e-14).detach()  # bn -> b
    
    return weight * torch.mean(reduced_batch_ce_loss + reduced_batch_dice_loss), weight * unreduced_batch_dice_loss, weight * unreduced_batch_ce_loss

