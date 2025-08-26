import subprocess
import yaml
import random
import numpy as np
import torch
from copy import deepcopy
from datetime import datetime
import os


def set_random_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False # 尽可能提升性能用True，可重复训练速度提高用False
    torch.backends.cudnn.deterministic = True

def get_timestamp():
    return datetime.now().strftime("%y%m%d-%H%M%S")

def get_commit_hash():
    message = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"])
    return message.strip().decode("utf-8")

def load_hparam(filename):
    stream = open(filename, "r", encoding = 'utf-8')
    docs = yaml.load_all(stream, Loader=yaml.Loader)
    hparam_dict = DotDict()
    for doc in docs:
        for k, v in doc.items():
            hparam_dict[k] = DotDict(v)
    return hparam_dict

class random_CutMix():
    def __init__(self):
        super().__init__()

    def CutMix(self, img, label, alpha = 1.0, force = False):
        assert label is not None
        if force:
            while True:
                indices = torch.randperm(img.size(0))
                shuffled_img = img[indices]
                shuffled_label = label[indices]
                catted_label = torch.cat([label, shuffled_label], dim = -1)
                finded = False
                for i in range(img.size(0)):
                    label_ = catted_label[i]
                    if (1 in label_ and 2 in label_) or (3 in label_ and 4 in label_):
                        finded = True
                        break
                if finded == True:
                    break
        else:
            indices = torch.randperm(img.size(0))
            shuffled_img = img[indices]
            shuffled_label = label[indices]

        lam = np.random.beta(alpha, alpha)
        lam = max(lam, 1 - lam)
        lam = min(0.75, lam)
        
        bbx1, bbx2, bby1, bby2 = self.rand_bbox(img.size(), lam)
        lam_ = 1 - (((bbx2 - bbx1) * (bby2 - bby1)) / (img.size(-1) * img.size(-2)))

        if lam_ > 0.75:
            shuffled_label = label
        else:
            img[: ,: ,bbx1: bbx2, bby1: bby2] = shuffled_img[: ,: ,bbx1: bbx2, bby1: bby2]

        return img, label, shuffled_label, lam_

    def rand_bbox(self, img_size, lam):
        W, H = img_size[-2:]
        cut_rat = np.sqrt(1 - lam)
        cut_w, cut_h = W * cut_rat, H * cut_rat

        cx = np.random.randint(cut_w // 2, W - cut_w // 2)
        cy = np.random.randint(cut_h // 2, H - cut_h // 2)

        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bby2 = np.clip(cy + cut_h // 2, 0, H)

        return int(bbx1), int(bbx2), int(bby1), int(bby2)

class DotDict(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __init__(self, dict_=None):
        super().__init__()
        if dict_ is not None:
            if not isinstance(dict_, dict):
                print(dict_)
                raise ValueError
            for k, v in dict_.items():
                if isinstance(v, dict):
                    self[k] = DotDict(v)
                else:
                    self[k] = v

    def __copy__(self):
        copy = type(self)()
        for k, v in self.items():
            copy[k] = v
        return copy

    def __deepcopy__(self, memodict={}):
        copy = type(self)()
        memodict[id(self)] = copy
        for k, v in self.items():
            copy[k] = deepcopy(v, memodict)
        return copy

    def __getstate__(self):
        return self.to_dict()

    def __setstate__(self, state):
        self.__init__(state)

    def to_dict(self):
        output_dict = dict()
        for k, v in self.items():
            if isinstance(v, DotDict):
                output_dict[k] = v.to_dict()
            else:
                output_dict[k] = v
        return output_dict
