import torch
import torch.nn as nn
import numpy as np

class AdaMix3D_boud(nn.Module):
    def __init__(self,
                 total_steps=0,
                 num_classes=4,
                 image_size=[96, 96, 96],
                 patch_size=8,
                 topk=16,
                 age=0,
                 self_paced=True,
                 device='cuda',
                 inverse=False):

        super(AdaMix3D_boud, self).__init__()
        self.device = device

        # patch grid
        self.d = image_size[0] // patch_size
        self.h = image_size[1] // patch_size
        self.w = image_size[2] // patch_size

        self.patch_size = patch_size

        self.age = age
        self.num_classes = num_classes
        self.topk = topk
        self.total_steps = total_steps
        self.self_paced = self_paced
        self.inverse = inverse

    # =========================================================

    # =========================================================
    def unfold_3d(self, x):
        """
        x: (B, C, D, H, W)
        return: (B, C, d, h, w, P)
        """
        B, C, D, H, W = x.shape
        p = self.patch_size

        x = x.view(
            B, C,
            self.d, p,
            self.h, p,
            self.w, p
        )

        x = x.permute(0, 1, 2, 4, 6, 3, 5, 7)

        return x.contiguous().view(B, C, self.d, self.h, self.w, p * p * p) # B,C,d,h,w,P

    # =========================================================

    # =========================================================
    def fold_3d(self, x):
        """
        x: (B, C, d, h, w, P)
        return: (B, C, D, H, W)
        """
        B, C, d, h, w, P = x.shape
        p = self.patch_size

        x = x.view(
            B, C,
            d, h, w,
            p, p, p
        )

        x = x.permute(0, 1, 2, 5, 3, 6, 4, 7)

        return x.contiguous().view(
            B, C,
            d * p,
            h * p,
            w * p
        )

    # =========================================================

    # =========================================================
    def dice_loss(self, prediction, target):
        """Calculating the dice loss
        Args:
            prediction = predicted image
            target = Targeted image
        Output:
            dice_loss"""
        target = torch.nn.functional.one_hot(target, num_classes=self.num_classes).permute(0, 4, 1, 2, 3).contiguous()
        smooth = 1e-5
        prediction = torch.softmax(prediction, dim=1)
        batchsize = target.size(0)
        # Calculate the Dice Similarity Coefficient for each class
        intersection = torch.sum(prediction * target, dim=(2, 3, 4))
        union = torch.sum(prediction + target, dim=(2, 3, 4))
        dice = ((2 * intersection) / (union + smooth)).mean(1)
        dice_loss = 1. - dice
        return dice_loss

    def increase_age(self, cur_step, total_steps):
        with torch.no_grad():
            self.age = self.sigmoid_rampup(cur_step, total_steps)

    def spl_curriculum(self, super_loss):
        m = super_loss < self.age
        v = m.clone().float()
        v = 1. - (super_loss / (self.age + 1e-5))
        return {'mask':m.tolist(), 'weight': v.tolist()}
    
    def sigmoid_rampup(self, current, rampup_length):
        """Exponential rampup from https://arxiv.org/abs/1610.02242"""
        if rampup_length == 0:
            return 1.0
        else:
            current = np.clip(current, 0.0, rampup_length)
            phase = 1.0 - current / rampup_length
            return float(np.exp(-5.0 * phase * phase))
    # =========================================================
    @torch.no_grad()
    def forward(self,
                oimage, aimage,
                olabel, alabel,
                oconf, aconf,
                prediction,
                cur_step):

        if self.self_paced:
            self.super_loss = self.dice_loss(prediction, olabel)
            splc = self.spl_curriculum(self.super_loss)
            sp_mask, sp_weight = splc['mask'], splc['weight']
        else:
            batch_size = oimage.shape[0]
            if self.inverse:
                sp_mask = torch.ones(batch_size).bool().tolist()  # [(True) * batch_size]
            else:
                sp_mask = torch.zeros(batch_size).bool().tolist()  # [(False) * batch_size]
            
            sp_weight = torch.ones(batch_size).tolist()

        oconf_map = oconf.clone().unsqueeze(1)
        aconf_map = aconf.clone().unsqueeze(1)

        B, C = oimage.shape[:2]

        # =====================================================

        # =====================================================
        oconf_unfolds = self.unfold_3d(oconf_map)
        aconf_unfolds = self.unfold_3d(aconf_map)

        oimage_unfolds = self.unfold_3d(oimage)
        aimage_unfolds = self.unfold_3d(aimage)

        olabel_unfolds = self.unfold_3d(olabel.unsqueeze(1).float())
        alabel_unfolds = self.unfold_3d(alabel.unsqueeze(1).float())
        
        label_min = olabel_unfolds.amin(dim=(2, 3, 4))
        label_max = olabel_unfolds.amax(dim=(2, 3, 4))
        boundary_mask = (label_min != label_max).squeeze(1)
        
        # mean confidence
        oconf_mean = oconf_unfolds.mean(dim=(1, 2, 3, 4))
        aconf_mean = aconf_unfolds.mean(dim=(1, 2, 3, 4))

        # =====================================================
        # patch mixing (unchanged)
        # =====================================================
        tok_list = []
        for i in range(B):

            topk = min(self.topk, abs(int(self.topk * sp_weight[i])))
            tok_list.append(topk)
            
            valid_mask = ~boundary_mask[i]
            valid_idx = torch.where(valid_mask)[0]

            valid_conf_o = oconf_mean[i][valid_idx]
            valid_conf_a = aconf_mean[i][valid_idx]

            _, order_o = torch.sort(valid_conf_o, descending=sp_mask[i])
            _, order_a = torch.sort(valid_conf_a, descending=not sp_mask[i])

            idx_o = valid_idx[order_o]
            idx_a = valid_idx[order_a]

            sel_o = idx_o[:topk]
            sel_a = idx_a[:topk]

            oimage_unfolds[i, :, :, :, :, sel_o] = aimage_unfolds[i, :, :, :, :, sel_a]
            olabel_unfolds[i, :, :, :, :, sel_o] = alabel_unfolds[i, :, :, :, :, sel_a]
            oconf_unfolds[i, :, :, :, :, sel_o] = aconf_unfolds[i, :, :, :, :, sel_a]

        # =====================================================

        # =====================================================
        oimage = self.fold_3d(oimage_unfolds).detach()
        olabel = self.fold_3d(olabel_unfolds).squeeze(1).long().detach()
        oconf = self.fold_3d(oconf_unfolds).squeeze(1).detach()

        self.increase_age(cur_step, self.total_steps)

        return oimage, olabel, oconf, self.super_loss, tok_list