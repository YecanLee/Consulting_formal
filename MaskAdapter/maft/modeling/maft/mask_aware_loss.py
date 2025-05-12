import torch
from torch import nn, Tensor
from torch.nn import functional as F

class MA_Loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.sl1 = nn.SmoothL1Loss()

    def forward(self, out_vocab_cls_results, mask_results, targets):
        '''
        input:  cls_score (out_vocab_cls_results)      bs * 100 * 172; 
                mask proposals (mask_results)          bs * 100 * h * w
                groundtruth (targets)                  {'labels': 1 * k; 'masks': k * h * w}
        
        output: ma_loss
        '''

        # Softmax to get class probabilities
        logits_per_image = F.softmax(out_vocab_cls_results[...,:-1], dim=-1)  # shape: (bs, 100, 171)

        total_loss = 0
        total_count = 0

        mask_results = mask_results.sigmoid()  # Ensure mask proposals are in the range [0, 1]

        for b in range(len(targets)):
            maski = mask_results[b].unsqueeze(0)  # shape: (1, 100, h, w)

            for i in range(targets[b]['masks'].shape[0]):
                logiti = logits_per_image[b, :, targets[b]['labels'][i]].unsqueeze(0)  # shape: (1, 100)
                labeli = targets[b]['masks'][i].unsqueeze(0)  # shape: (1, h, w)

                # Compute IoU directly for current instance
                iou = self.get_iou(maski, labeli).detach()  # shape: (1, 100)

                # Compute loss for current instance
                loss = self.sl1(logiti, iou)
                total_loss += loss
                total_count += 1

        # Average the loss over the number of instances
        ma_loss = total_loss / total_count if total_count > 0 else 0
        return ma_loss

    def get_iou(self, pred, target):
        b, c, h, w = pred.shape
        if len(target.shape) != len(pred.shape):
            target = target.unsqueeze(1) 

        if pred.shape[-2:] != target.shape[-2:]:
            pred = F.interpolate(pred, size=(target.shape[-2], target.shape[-1]), mode="bilinear", align_corners=False)

        pred = pred.reshape(b, c, -1)
        target = target.reshape(b, 1, -1)
        
        # Compute the IoU of the foreground
        Iand1 = torch.sum(target * pred, dim=-1)
        Ior1 = torch.sum(target, dim=-1) + torch.sum(pred, dim=-1) - Iand1 + 1e-7
        IoU1 = Iand1 / Ior1

        return IoU1

    def mynorm(self, embeding):
        assert len(embeding.shape) == 2, embeding.shape
        min_em, _ = torch.min(embeding, dim = -1)
        max_em, _ = torch.max(embeding, dim = -1)
        embeding = (embeding-min_em.unsqueeze(-1))/((max_em-min_em+0.00000001).unsqueeze(-1))
        return embeding


