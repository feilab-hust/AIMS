import torch
import torch.nn as nn
import numpy as np
from loss.sub_loss import *

class loss_used(nn.Module):
    def __init__(self, hp):
        super().__init__()
        self.loss_cfg = hp.train.loss

        # self.loss_dict = {'crossentropy': LOSS_REGISTRY.get('CE'),
        #                   'l1': LOSS_REGISTRY.get('l1'),
        #                   'l2': LOSS_REGISTRY.get('l2'),
        #                   }

        self.name = self.loss_cfg.name

    def forward(self, output: dict, input, GT, GT_grad, Lam = 1):
        inf = output['output']
        inf_img = output['img_output']
        latent_code = output['latent_code']
        loss = 0
        loss_state = {}
        latent_supContrast = None

        if latent_code is not None and len(latent_code) == len(GT) * 2:
            # latent_1, latent_2 = torch.chunk(latent_code, chunks = 2, dim = 0)
            latent_1, latent_2 = torch.split(latent_code, [len(GT), len(GT)], dim = 0)
            latent_supContrast = torch.stack([latent_1, latent_2],dim = 1)

        for n in self.name:
            if n == 'grad':
                inf_grad = output['output_grad']
                sub_loss_fn = LOSS_REGISTRY.get('l2')(reduction = self.loss_cfg[n]['reduction'])
                sub_loss = sub_loss_fn(inf_grad, GT_grad).mean(2)
            elif n == 'scl':
                if latent_supContrast is not None:
                    sub_loss_fn = LOSS_REGISTRY.get(str(n).lower())(reduction = self.loss_cfg[n]['reduction'])
                    sub_loss = sub_loss_fn(latent_supContrast, GT)
                else:
                    continue
            elif n in ['l2', 'smooth_l1']:
                sub_loss_fn = LOSS_REGISTRY.get(str(n).lower())(reduction = self.loss_cfg[n]['reduction'])
                sub_loss = sub_loss_fn(inf_img, input)
            elif n == 'fcl':
                sub_loss_fn = LOSS_REGISTRY.get(str(n).lower())(reduction = self.loss_cfg[n]['reduction'])
                sub_loss = sub_loss_fn(inf, GT, Lam)
            else:
                # sub_loss_fn = self.loss_dict[n](**(self.loss_cfg[n]))
                sub_loss_fn = LOSS_REGISTRY.get(str(n).lower())(reduction = self.loss_cfg[n]['reduction'])
                sub_loss = sub_loss_fn(inf, GT)

            # sub_loss_weight = sub_loss.mean(dim = [-1, -2], keepdim = True).detach() if n != 'ssim' else sub_loss
            # sub_loss_weight = sub_loss_weight / sub_loss_weight.mean()
            loss = loss + self.loss_cfg[n]['weight'] * (sub_loss).mean() # * sub_loss_weight ** 2.
            loss_state.update({str(n): 1 - sub_loss.mean().detach().item() if n == 'ssim' else sub_loss.mean().detach().item()})

        return loss, loss_state








