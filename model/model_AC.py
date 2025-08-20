import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, MultiStepLR
from timm.scheduler.cosine_lr import CosineLRScheduler
import numpy as np

from collections import OrderedDict
from pathlib2 import Path

from util.util import DotDict


class Model:
    def __init__(self, hp_net, net_arch, loss_f, rank=0, world_size=1):
        self.hp_net = hp_net
        self.device = self.hp_net.model.device
        self.net = net_arch.to(self.device)
        self.rank = rank
        self.world_size = world_size
        if self.device != "cpu" and self.world_size != 0:
            self.net = DDP(self.net, device_ids=[self.rank])
        self.input = None
        self.GT = None
        self.step = 0
        self.sub_step = 0
        self.epoch = -1

        self.n_epoch = hp_net.train.num_epoch

        # init optimizer
        optimizer_mode = self.hp_net.train.optimizer.mode
        if optimizer_mode == "adam":
            self.optimizer = torch.optim.Adam(
                self.net.parameters(), **(self.hp_net.train.optimizer['param'])
            )
        elif optimizer_mode == 'adamw':
            self.optimizer = torch.optim.AdamW(
                self.net.parameters(), **(self.hp_net.train.optimizer['param'])
            )
        elif optimizer_mode == 'sgd_nesterov':
            self.optimizer = torch.optim.SGD(
                self.net.parameters(), lr = self.hp_net.train.optimizer['param']['lr'],
                momentum = 0.8, nesterov = True
            )
        else:
            raise Exception("%s optimizer not supported" % optimizer_mode)

        # init optimizer lr_scheduler
        noise_args = dict(
            noise_range_t=None,
            noise_pct=0.67,
            noise_std=1.,
            noise_seed=42,
        )
        cycle_args = dict(
            cycle_mul=1.,
            cycle_decay=1,
            cycle_limit=1,
        )        
        scheduler = self.hp_net.train.optimizer.scheduler
        scheduler_mode = scheduler.mode
        if scheduler_mode == 'CosineAnnealingWarm':
            self.lr_scheduler = CosineAnnealingWarmRestarts(
                self.optimizer, T_0 = scheduler.T_0,
                T_mult = scheduler.T_mult,
                eta_min = scheduler.min_lr_ratio * self.hp_net.train.optimizer['param']['lr'])
        elif scheduler_mode == 'MultiStep':
            self.lr_scheduler = MultiStepLR(
                self.optimizer, milestones = scheduler.decay_step, gamma = scheduler.decay_gamma)
        elif scheduler_mode == 'CosineLRScheduler':
            self.lr_scheduler = CosineLRScheduler(
                self.optimizer, t_initial = scheduler.T_0,
                lr_min = scheduler.min_lr_ratio * self.hp_net.train.optimizer['param']['lr'],
                warmup_lr_init = scheduler.warmup_lr,
                warmup_t = scheduler.warmup_epoch,
                k_decay = 1.0, t_in_epochs = True,
                **cycle_args, **noise_args,
            )
        else:
            self.lr_scheduler = None

        # init loss
        self.loss_f = loss_f
        self.log = DotDict()
        # init lr
        self.log.lr = self.optimizer.param_groups[0]['lr']

    def feed_data(self, **data):  # data's keys: input, GT
        for k, v in data.items():
            if k not in ['GT_sample', 'Lam'] and data[k] is not None:
                data[k] = v.to(self.device)
            else:
                data[k] = v
        self.input = data.get("input")
        self.GT = data.get("GT")
        self.GT_grad = data.get('GT_grad')
        self.GT_sample = data.get('GT_sample')
        self.Lam = data.get('Lam')

    def optimize_parameters(self):
        self.net.train()
        self.optimizer.zero_grad()
        self.output = self.run_network()

        if len(self.output['output']) == len(self.GT) * 2 and self.GT.ndim == 1:
            self.output['output'] = torch.split(self.output['output'],[len(self.GT), len(self.GT)], dim = 0)[0]
            if self.output['img_output'] is not None:
                self.output['img_output'] = torch.split(self.output['img_output'],[len(self.GT), len(self.GT)], dim = 0)[0]

            self.input_interp = F.interpolate(self.input, scale_factor= (0.25, 0.25), mode = 'bilinear')
            loss_v, loss_v_state = self.loss_f(self.output,
                                            torch.split(self.input_interp,[len(self.GT), len(self.GT)], dim = 0)[0],
                                            self.GT, self.GT_grad)
        elif self.GT.ndim > 1:
            loss_v_0, loss_v_state_0 = self.loss_f(self.output, None, self.GT[0], self.GT_grad, self.Lam)
            loss_v_1, loss_v_state_1 = self.loss_f(self.output, None, self.GT[1], self.GT_grad, self.Lam)
            loss_v = loss_v_0 * self.Lam + loss_v_1 * (1 - self.Lam)
            for k, v in loss_v_state_0.items():
                loss_v_state_0[k] = v * self.Lam + loss_v_state_1[k] * (1 - self.Lam)
            loss_v_state = loss_v_state_0
        else:
            loss_v, loss_v_state = self.loss_f(self.output, None, self.GT, self.GT_grad)


        loss_v.backward()
        self.optimizer.step()

        # set log
        self.log.loss_v = loss_v.item()
        gt = self.GT[0] if self.GT.ndim > 1 else self.GT
        self.log.acc_v = ((F.softmax(self.output['output'], dim = -1).argmax(dim = 1) == gt) * 1.0).mean().detach().item()
        self.log.loss_v_state = loss_v_state

    def run_lr_scheduler(self):
        if self.lr_scheduler:
            self.lr_scheduler.step() 
            # set log
            self.log.lr = self.optimizer.param_groups[0]['lr']
        else:
            pass

    def model_test(self, dual_class = False):
        self.output_evidential = None
        self.output = self.inference()

        ## index for inner data
        inner_index = list(filter(lambda i: self.GT[i] in [0, 1, 2, 3, 4], range(len(self.GT))))
        if inner_index and self.output['latent_code'] is not None: 
            latent_1, latent_2 = torch.split(self.output['latent_code'], [len(self.GT), len(self.GT)], dim = 0)
            output_dict = dict(
                output = torch.split(self.output['output'], [len(self.GT), len(self.GT)], dim = 0)[0][inner_index],
                latent_code = torch.cat([latent_1[inner_index], latent_2[inner_index]], dim = 0),
                img_output = None
            )
            _, evaluate_loss_v_state = self.loss_f(output_dict, None, self.GT[inner_index].type(torch.long), self.GT_grad)

            acc_evaluate = ((self.output['output'][inner_index].argmax(dim = 1) == self.GT[inner_index]) * 1.0).mean().item()
        else:
            output_dict = dict(
                output = self.output['output'][inner_index],
                latent_code = None,
                img_output = None
            )
            _, evaluate_loss_v_state = self.loss_f(output_dict, None, self.GT[inner_index].type(torch.long), self.GT_grad)
            if not dual_class:
                acc_evaluate = ((self.output['output'][inner_index].argmax(dim = 1) == self.GT[inner_index]) * 1.0).mean().item()
            else:
                if self.GT[inner_index] == 0:
                    acc_evaluate = ((self.output['output'][inner_index].argmax(dim = 1) == self.GT[inner_index]) * 1.0).mean().item()
                else:
                    acc_evaluate = ((self.output['output'][inner_index].argmax(dim = 1) != 0) * 1.0).mean().item()
        return self.output, self.output_evidential, evaluate_loss_v_state, acc_evaluate

    def inference(self):
        self.net.eval()
        return self.run_network()

    def inference_bayesian(self, bayesian_num = 8): 
        self.net.eval()
        output_ = self.run_network()

        for module in self.net.modules():
            if module.__class__.__name__.startswith('Dropout'):
                module.train()

        output_list = []
        latent_code_list = []

        for _ in range(bayesian_num):
            output = self.run_network()
            output_list.append(output['output'])
            latent_code_list.append(output['latent_code'])

        output_list_tensor = torch.stack(output_list, dim = 0)
        output_mean = output_['output'] 
        output_std = torch.std(F.softmax(output_list_tensor, dim = -1), dim = 0)

        if latent_code_list[0] is not None:    
            latent_code_list_tensor = torch.stack(latent_code_list, dim = 0)
            latent_code_mean = latent_code_list_tensor[0] 
            latent_code_std = torch.std(latent_code_list_tensor, dim = 0)
        else:
            latent_code_mean = None
            latent_code_std = None

        return {'img_output': None, 'output': output_mean, 'output_std': output_std,
                'kl_d': None, 'latent_code': latent_code_mean, 'latent_code_std': latent_code_std}


    def run_network(self):

        label_output, letent_code = self.net(self.input)
        img_output = None
        kl_d = None

        return {'img_output': img_output, 'output': label_output, 'kl_d': kl_d, 'latent_code': letent_code}

    def save_network(self, logger, save_file=True):
        if self.rank == 0:
            net = self.net.module if isinstance(self.net, DDP) else self.net
            state_dict = net.state_dict()
            for key, param in state_dict.items():
                state_dict[key] = param.to("cpu")
            if save_file:
                network_path = Path(self.hp_net.log.chkpt_dir) / 'network'
                network_path.mkdir(parents = True, exist_ok = True)
                save_filename = "%s_epoch_%d_step_%d.pth" % (self.hp_net.log.name, self.epoch, self.step)
                save_path = network_path / save_filename
                torch.save(state_dict, str(save_path))
                if logger is not None:
                    logger.info("Saved network checkpoint to: %s" % save_path)
            return state_dict

    def load_network(self, loaded_net=None, logger=None, ckpt = None):
        add_log = False
        if loaded_net is None:
            add_log = True
            if ckpt == None:
                ckpt = self.hp_net.load.network_chkpt_path
            loaded_net = torch.load(
                ckpt, map_location=torch.device(self.device)
            )
        loaded_clean_net = OrderedDict()  # remove unnecessary 'module.'
        for k, v in loaded_net.items():
            if k.startswith("module."):
                loaded_clean_net[k[7:]] = v
            else:
                loaded_clean_net[k] = v
        self.net.load_state_dict(loaded_clean_net, strict=self.hp_net.load.strict_load)
        if logger is not None and add_log:
            logger.info("Checkpoint %s is loaded" % self.hp_net.load.network_chkpt_path)

    def reshape_transform(self, tensor, h = 16, w = 16):
        result = tensor[:, 1:, :].reshape(tensor.size(0), h, w, tensor.size(-1))
        result = result.permute(0, 3, 1, 2)
        return result

    def norm(self, x: np.ndarray):
        x = x - x.min()
        x = x / x.max()
        return x.astype(np.float32)
