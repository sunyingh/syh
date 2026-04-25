from __future__ import absolute_import, division, print_function

import random

import numpy as np
import torch
import torch.nn.functional as F

from options_PCC import MonodepthOptions
from trainer_end_to_end import Trainer as BaseTrainer
from utils.layers import get_smooth_bright, get_smooth_loss, ncc_loss


class Trainer(BaseTrainer):
    """Trainer with PCC loss added on top of the current Ubuntu EndoDAC trainer."""

    def __init__(self, options):
        super().__init__(options)
        self.pcc_enabled = getattr(self.opt, "use_pcc", False)
        self.pcc_window = max(1, int(getattr(self.opt, "pcc_window", 5)))

        if self.pcc_enabled:
            print(
                "PCC enabled: "
                f"pcc_weight={self.opt.pcc_weight}, "
                f"cycle={self.opt.pcc_use_cycle}, "
                f"cycle_weight={self.opt.pcc_cycle_weight}, "
                f"window={self.pcc_window}"
            )

    def local_normalize(self, x):
        """Normalize local contrast to reduce brightness sensitivity."""
        pad = self.pcc_window // 2
        mean = F.avg_pool2d(x, self.pcc_window, stride=1, padding=pad)
        sq_mean = F.avg_pool2d(x * x, self.pcc_window, stride=1, padding=pad)
        var = torch.clamp(sq_mean - mean * mean, min=1e-6)
        return (x - mean) / torch.sqrt(var)

    @staticmethod
    def masked_mean(value_map, mask):
        return (value_map * mask).sum() / (mask.sum() + 1e-7)

    def invariant_ncc_loss(self, pred, target, mask):
        pred_gray = pred.mean(1, keepdim=True)
        target_gray = target.mean(1, keepdim=True)

        pred_norm = self.local_normalize(pred_gray)
        target_norm = self.local_normalize(target_gray)

        loss_map = 1.0 + ncc_loss(
            pred_norm,
            target_norm,
            win=[self.pcc_window, self.pcc_window],
        )
        return self.masked_mean(loss_map, mask)

    def compute_pcc_loss(self, inputs, outputs, scale, frame_id):
        target = inputs[("color", 0, 0)]
        source = inputs[("color", frame_id, 0)]
        occu_mask = outputs[("occu_mask_backward", 0, frame_id)].detach()

        registration = outputs[("registration", scale, frame_id)]
        refined = outputs[("refined", scale, frame_id)]

        loss = self.invariant_ncc_loss(registration, target, occu_mask)
        loss += self.invariant_ncc_loss(refined, target, occu_mask)

        if getattr(self.opt, "pcc_use_cycle", False):
            cycle_back = self.spatial_transform(
                registration,
                outputs[("position_reverse", "high", scale, frame_id)],
            )
            loss += self.opt.pcc_cycle_weight * self.invariant_ncc_loss(
                cycle_back,
                source,
                occu_mask,
            )

        return loss

    def compute_losses(self, inputs, outputs):
        losses = {}
        total_loss = 0

        for scale in self.opt.scales:
            loss = 0
            loss_reprojection = 0
            loss_transform = 0
            loss_cvt = 0
            loss_pcc = 0

            disp = outputs[("disp", scale)]
            color = inputs[("color", 0, scale)]

            for frame_id in self.opt.frame_ids[1:]:
                occu_mask_backward = outputs[("occu_mask_backward", 0, frame_id)].detach()

                loss_reprojection += self.masked_mean(
                    self.compute_reprojection_loss(
                        outputs[("color", frame_id, scale)],
                        outputs[("refined", scale, frame_id)],
                    ),
                    occu_mask_backward,
                )
                loss_transform += self.masked_mean(
                    torch.abs(
                        outputs[("refined", scale, frame_id)]
                        - outputs[("registration", 0, frame_id)].detach()
                    ).mean(1, True),
                    occu_mask_backward,
                )
                loss_cvt += get_smooth_bright(
                    outputs[("transform", "high", scale, frame_id)],
                    inputs[("color", 0, 0)],
                    outputs[("registration", scale, frame_id)].detach(),
                    occu_mask_backward,
                )

                if self.pcc_enabled:
                    loss_pcc += self.compute_pcc_loss(inputs, outputs, scale, frame_id)

            mean_disp = disp.mean(2, True).mean(3, True)
            norm_disp = disp / (mean_disp + 1e-7)
            smooth_loss = get_smooth_loss(norm_disp, color)

            loss += loss_reprojection / 2.0
            loss += self.opt.transform_constraint * (loss_transform / 2.0)
            loss += self.opt.transform_smoothness * (loss_cvt / 2.0)
            loss += self.opt.disparity_smoothness * smooth_loss / (2 ** scale)

            if self.pcc_enabled:
                loss += self.opt.pcc_weight * (loss_pcc / 2.0)
                losses["loss_pcc/{}".format(scale)] = loss_pcc / 2.0

            total_loss += loss
            losses["loss/{}".format(scale)] = loss

        total_loss /= self.num_scales
        losses["loss"] = total_loss
        return losses


def random_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


if __name__ == "__main__":
    options = MonodepthOptions()
    opts = options.parse()
    random_seeds(314)
    trainer = Trainer(opts)
    trainer.train()
