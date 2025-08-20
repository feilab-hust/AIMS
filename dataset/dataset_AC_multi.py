from torch.utils.data import Dataset, DataLoader
import torch
from torchvision import transforms
from dataset.utils import DataloaderMode
from pathlib2 import Path
import numpy as np
import cv2
from PIL import Image
import random
from torchvision import datasets
import tifffile as tiff
from tqdm import tqdm
import kornia
import pywt

class2label = {
    'Young': 0, 
    'Doxorubicin': 1, 
    'Etoposide': 2, 
    'Antimycin': 3, 
    'Oxidative stress': 4
    }

class Dataset_AC_multi(Dataset):
    def __init__(self, hp, mode, logger):
        super().__init__()
        self.hp = hp
        self.mode_s = mode
        self.path_data_mit = hp.data.train_dir_mit_multi
        self.path_data_nuc = hp.data.train_dir_nuc_multi
        self.orga = hp.data.organelle
        self.mode = hp.data.mode
        test_path_dir = hp.data.test_dir_multi
        self.half_size = hp.data.half_size
        img_size = hp.data.img_size if self.half_size else hp.train.image_size
        self.test_idx = [53, 86, 6, 45, 89, 56, 17, 0, 25, 67] # 
        self.balanced = hp.data.balanced
        self.img_edge = hp.data.img_edge
        self.class_list  = hp.data.class_list
        self.logger = logger
        self.key_name = hp.data.key_name
        self.num_constant = hp.data.num_constant
        self.eval_on = hp.data.whether_eval

        path_1102_mit = Path(self.path_data_mit)
        path_1102_nuc = Path(self.path_data_nuc)

        ''' 无testdir的情况：预先保存成txt还是重新索引 '''
        if hp.data.test_dir_multi is None and \
            (path_1102_mit / f'{self.mode}_{self.orga}_Old_Train.txt').exists() and \
            (path_1102_mit / f'{self.mode}_{self.orga}_Old_Test.txt').exists() and \
            (path_1102_mit / f'{self.mode}_{self.orga}_Young_Train.txt').exists() and \
            (path_1102_mit / f'{self.mode}_{self.orga}_Young_Test.txt').exists():

            pass #TODO: 先不加入读取过程

            # self.read_txt2list(AC_list_train, f'{self.mode}_{self.orga}_Old_Train.txt', '1.0')
            # self.read_txt2list(AC_list_train, f'{self.mode}_{self.orga}_Young_Train.txt', '0')
            # self.read_txt2list(AC_list_test, f'{self.mode}_{self.orga}_Old_Test.txt', '1.0')
            # self.read_txt2list(AC_list_test, f'{self.mode}_{self.orga}_Young_Test.txt', '0')

        else:
            if hp.data.test_dir_multi == True:
                AC_dict_multi_test_nuc, _ = self.get_path_list_multi_calsses(Path(hp.data.test_dir_nuc_multi), key_name = self.key_name, class_name_list = self.class_list, orga = 'nucleus', is_mask = True if hp.data.organelle == 'mask' else False, excel_ = hp.data.excel_nuc_train)
                AC_dict_multi_test_mit, _ = self.get_path_list_multi_calsses(Path(hp.data.test_dir_mit_multi), key_name = self.key_name, class_name_list = self.class_list, orga = 'mito', is_mask = True if hp.data.organelle == 'mask' else False, excel_ = hp.data.excel_mito_train)

                self.sort_by_idx(AC_dict_multi_test_nuc)
                self.sort_by_idx(AC_dict_multi_test_mit)

            if hp.data.eval_dir_multi == True:
                AC_dict_multi_eval_nuc, _ = self.get_path_list_multi_calsses(Path(hp.data.eval_dir_nuc_multi), key_name = self.key_name, class_name_list = self.class_list, orga = 'nucleus', is_mask = True if hp.data.organelle == 'mask' else False, excel_ = hp.data.excel_nuc_train)
                AC_dict_multi_eval_mit, _ = self.get_path_list_multi_calsses(Path(hp.data.eval_dir_mit_multi), key_name = self.key_name, class_name_list = self.class_list, orga = 'mito', is_mask = True if hp.data.organelle == 'mask' else False, excel_ = hp.data.excel_mito_train)

                self.sort_by_idx(AC_dict_multi_eval_nuc)
                self.sort_by_idx(AC_dict_multi_eval_mit)

            AC_dict_multi_nuc, class_num_list_nuc = self.get_path_list_multi_calsses(path_1102_nuc, key_name = self.key_name, class_name_list = self.class_list, orga = 'nucleus', is_mask = True if hp.data.organelle == 'mask' else False, excel_ = hp.data.excel_nuc_train)
            AC_dict_multi_mit, class_num_list_mit = self.get_path_list_multi_calsses(path_1102_mit, key_name = self.key_name, class_name_list = self.class_list, orga = 'mito', is_mask = True if hp.data.organelle == 'mask' else False, excel_ = hp.data.excel_mito_train)

            self.sort_by_idx(AC_dict_multi_mit)
            self.sort_by_idx(AC_dict_multi_nuc)

            # min_train_num_mit = min(class_num_list_mit)
            min_train_num_mit = min_train_num_nuc = 0.90 # 0.5 0.95 0.90
            # min_train_num_nuc = min(class_num_list_nuc)
            if hp.data.test_dir_multi == False:
                AC_dict_multi_train_mit = self.extract_train_dict(AC_dict_multi_mit, min_train_num_mit, balanced = self.balanced, num_constant=self.num_constant)
                AC_dict_multi_eval_mit, AC_dict_multi_test_mit = self.extract_test_dict(AC_dict_multi_mit, AC_dict_multi_train_mit, whether_eval = self.eval_on, val_num = hp.data.val_num, excel_ = hp.data.excel_mito_val_test)
                AC_dict_multi_train_nuc = self.extract_train_dict(AC_dict_multi_nuc, min_train_num_nuc, balanced = self.balanced, num_constant=self.num_constant)
                AC_dict_multi_eval_nuc, AC_dict_multi_test_nuc = self.extract_test_dict(AC_dict_multi_nuc, AC_dict_multi_train_nuc, whether_eval = self.eval_on, val_num = hp.data.val_num, excel_ = hp.data.excel_nuc_val_test)
            else:
                assert hp.data.eval_dir_multi == True, ValueError('eval_dir_multi should exist')

                AC_dict_multi_train_mit = AC_dict_multi_mit
                AC_dict_multi_train_nuc = AC_dict_multi_nuc

            # if hp.data.test_dir_multi is None:
            #     self.write_list2txt(AC_list_old_train, name = f'{self.mode}_{self.orga}_Old_Train')
            #     self.write_list2txt(AC_list_old_test, name = f'{self.mode}_{self.orga}_Old_Test')
            #     self.write_list2txt(AC_list_young_train, name = f'{self.mode}_{self.orga}_Young_Train')
            #     self.write_list2txt(AC_list_young_test, name = f'{self.mode}_{self.orga}_Young_Test')

            if hp.data.organelle == 'mito':
                self.print_data(AC_dict_multi_train_mit, AC_dict_multi_eval_mit, AC_dict_multi_test_mit, self.hp.data.save_excel, self.hp.data.excel_save_path, self.hp.log.name.split('_[')[-2])
            elif hp.data.organelle == 'nucleus':
                self.print_data(AC_dict_multi_train_nuc, AC_dict_multi_eval_nuc, AC_dict_multi_test_nuc, self.hp.data.save_excel, self.hp.data.excel_save_path, self.hp.log.name.split('_[')[-2])
            else:
                raise ValueError(f"invalid organelle {hp.data.organelle}")

        AC_list_multi_train_mit = self.dict2list(AC_dict_multi_train_mit)
        AC_list_multi_train_nuc = self.dict2list(AC_dict_multi_train_nuc)
        AC_list_multi_eval_mit = self.dict2list(AC_dict_multi_eval_mit)
        AC_list_multi_eval_nuc = self.dict2list(AC_dict_multi_eval_nuc)
        AC_list_multi_test_mit = self.dict2list(AC_dict_multi_test_mit)
        AC_list_multi_test_nuc = self.dict2list(AC_dict_multi_test_nuc)
        if mode is DataloaderMode.train:
            self.dataset_nuc = AC_list_multi_train_nuc
            self.dataset_mit = AC_list_multi_train_mit
        elif mode is DataloaderMode.eval:
            if self.eval_on:
                self.dataset_nuc = AC_list_multi_eval_nuc
                self.dataset_mit = AC_list_multi_eval_mit
            else:
                self.dataset_nuc = AC_list_multi_test_nuc
                self.dataset_mit = AC_list_multi_test_mit           
        elif mode is DataloaderMode.test:
            self.dataset_nuc = AC_list_multi_test_nuc
            self.dataset_mit = AC_list_multi_test_mit
        else:
            raise ValueError(f"invalid dataloader mode {mode}")
        
        # assert all([len(self.dataset_nuc[x]) == len(self.dataset_mit[x]) for x in self.dataset_nuc.keys()]), ValueError('nuc and mit dataset should have the same length')

        self.data_augment = transforms.Compose([transforms.Resize([img_size, img_size]), # 512
                                                transforms.RandomHorizontalFlip(p = 0.5),
                                                transforms.RandomVerticalFlip(p = 0.5),
                                                # transforms.RandomAffine(degrees = 0, translate = (0.05, 0.15)), 
                                               ])
        self.data_augment_affine = transforms.Compose([transforms.RandomAffine(degrees = 20, shear = 20)]) # 旋转导致的边缘填充可能会导致问题
        self.data_augment_rotate = transforms.Compose([transforms.RandomRotation(degrees = 20, expand = True),
                                                       ]) # 旋转导致的边缘填充可能会导致问题，导致图片倍率变化，可能会影响线粒体的形态分布
        self.data_trans = transforms.Compose([transforms.Resize([img_size, img_size])])

    def __len__(self):
        if self.hp.data.organelle == 'mito':
            return len(self.dataset_mit)
        elif self.hp.data.organelle == 'nucleus':
            return len(self.dataset_nuc)
        else:
            raise NotImplementedError

    def __getitem__(self, idx):
        if self.hp.data.organelle == 'mito':
            p_mit, label_mit = list(self.dataset_mit[idx].items())[0]
        elif self.hp.data.organelle == 'nucleus':
            p_nuc, label_nuc = list(self.dataset_nuc[idx].items())[0]
        else:
            raise ValueError(f"invalid organelle {self.hp.data.organelle}")


        if self.hp.data.organelle == 'nucleus':
            img = self.img_read(p_nuc)
            label_ = class2label[label_nuc.strip()]
        elif self.hp.data.organelle == 'mito':
            img = self.img_read(p_mit)
            label_ = class2label[label_mit.strip()]
        elif all(organelle in self.hp.data.organelle for organelle in ['mito', 'nucleus']):
            raise NotImplementedError('mito & nucleus mixed is not supported for multi_classification yet')
        else:
            raise ValueError(f"invalid organelle {self.hp.data.organelle}")

        
        if random.random() > 0.5 and self.mode_s == DataloaderMode.train and self.hp.data.organelle == 'mito':
            img = DataAugment()(img, type_ = 'blur')

        img_tensor = img
        
        if self.hp.data.organelle not in ['mask', 'nucleus']:

            if ' -t' not in self.mode:
                img_tensor = self.norm(img_tensor)
                img_tensor = torch.clamp(img_tensor, 0.0, 1.0)
            else:
                img_tensor[0] = self.norm(img_tensor[0])
                img_tensor[1] = self.norm(img_tensor[1])
                img_tensor = torch.clamp(img_tensor, 0.0, 1.0)
        else:
            img_tensor = img * 0.9 + 0.1


        if self.img_edge is not None:
            if self.img_edge == 'sobel':
                img_tensor = self.image_edge_sobel(img_tensor)
            elif self.img_edge == 'wavelet':
                img_tensor = self.image_edge_wavelet(img_tensor)
            else:
                raise ValueError()
        else:
            pass
        
        if self.hp.data.organelle not in ['mask', 'nucleus']:
            img_tensor = self.norm(img_tensor)


        if self.hp.data.dual_class and label_ > 0:
            label_ = 1.

        return [img_tensor, torch.LongTensor([float(label_)])]
    

    def image_edge_sobel(self, img_tensor):
        img_sobel = kornia.filters.spatial_gradient(img_tensor.unsqueeze(0), mode='sobel', order=1, normalized=True).squeeze(0)
        return img_sobel

    def image_edge_wavelet(self, img_tensor):
        img_wavelet_coeffs = pywt.dwt2(img_tensor, 'haar')
        LL, (LH, HL, HH) = img_wavelet_coeffs
        return torch.from_numpy(np.concatenate([LL, LH, HL, HH], axis = 0))

    def img_read(self, p_):
        if ' -t' in self.mode:
            assert len(p_) == 2

            imgs = np.stack([tiff.imread(p_[0]), tiff.imread(p_[1])], axis = 0)
            img_0 = imgs[0]
            grad = np.abs(imgs[1]-imgs[0])
            img = np.stack([img_0, grad], axis = 0)
        else:
            img = tiff.imread(str(p_))
            if img.ndim != 3:
                img = img[None, ...] 

        assert 'img' in locals() and img.ndim == 3, ValueError('img should have 3 dimensions')

        img = torch.from_numpy(img).type(torch.float32)
        if self.mode_s == DataloaderMode.train:
            img = self.data_augment(img)
        else:
            img = self.data_trans(img)

        return img
    
    def norm(self, x):
        x = x - x.min()
        x = x / x.max()
        return x

    def norm_per(self, x: torch.Tensor):
        x = x - torch.amin(x, dim = (-1,-2), keepdim = True)
        x = x / torch.amax(x, dim = (-1, -2), keepdim = True)
        return x

    def random_crop(self, img):
        H, W = img.shape[:2]
        if H > self.patch_size:
            xx = np.random.randint(0, H - self.patch_size)
            img = img[xx: xx + self.patch_size, :]
        if W > self.patch_size:
            yy = np.random.randint(0, W - self.patch_size)
            img = img[:, yy: yy + self.patch_size]
        return img

    def path_load(self, path, list_single_class, orga):
        try:
            for p_ in path.iterdir():
                if p_.suffix == '.tif' and orga in p_.parent.name:
                    list_single_class.append(p_)
                elif p_.is_dir() and p_.name != 'compare' and p_.name != 'generated_mask':
                    self.path_load(p_, list_single_class, orga)
                else:
                    pass
        except:
            pass

    def path_load_mask(self, path, p_list):
        try:
            for p_ in path.iterdir():
                if p_.suffix == '.tif' and 'generated_mask' in p_.parent.name:
                    p_list.append(p_)
                elif p_.is_dir() and p_.name != 'compare':
                    self.path_load_mask(p_, p_list)
                else:
                    continue
        except:
            raise ValueError 

    def write_list2txt(self, list_, name):
        p_txt = Path(self.path_data) / f'{name}.txt'
        if p_txt.exists():
            p_txt.unlink()
        with open(str(p_txt), 'w') as t:
            for p_ in list_:
                if isinstance(p_, tuple):
                    t.write(str([str(p_[i]) for i in range(len(p_))]) + '\n')
                else:
                    t.write(str(p_) + '\n')
        t.close()

    def read_txt2list(self, list, name, label):
        p_txt = Path(self.path_data) / name
        with open(str(p_txt), 'r') as t:
            for l_ in t:
                if l_.startswith('['):
                    p_0, p_1 = l_.split(', ')
                    p_ = (Path(p_0[2:-1]), Path(p_1[1:-3]))
                    list.append({p_: label})
                else:
                    list.append({Path(l_[:-1]): label})

    def sort_by_idx(self, dict_):

        for k in dict_.keys():
            dict_[k] = sorted(dict_[k], key = lambda y: int(y.name.split('_')[0]) if y.name.split('_')[0].isdigit() else int(y.name.split('_')[1]))

    def get_path_list_multi_calsses(self, path, key_name, class_name_list, orga = 'mito', is_mask = False, excel_ = False):
        dict_multi_classes = {}
        class_num_list = []
        if not excel_:
            for idx, p in enumerate(path.iterdir()):
                if p.is_file() or str(p.name).split(' ')[0] not in class_name_list: # key_name not in p.name or 
                    continue

                name = str(p.name).split('-')[0]
                if name.endswith(' '):
                    name = name[:-1]
                dict_ = {name: []}

                if not is_mask:
                    self.path_load(p / self.mode, dict_[name], orga)
                else:
                    self.path_load_mask(p / '3D-WF (single)', dict_[name])

                if name not in list(dict_multi_classes.keys()):
                    dict_multi_classes.update(dict_)
                else:
                    dict_multi_classes[name].update(dict_)

                class_num_list.append(len(dict_[name]))
        else:
            import pandas as pd
            assert self.hp.data.excel_path_train is not None and Path(self.hp.data.excel_path_train).exists(), ValueError('excel_path_train should be provided')
            sheets = pd.ExcelFile(self.hp.data.excel_path_train).sheet_names
            for sheet in sheets:
                df = pd.read_excel(self.hp.data.excel_path_train, sheet_name = sheet, usecols = [0], header = None)
                file_paths = df.values.tolist()
                for k, fp in enumerate(file_paths):
                    file_paths[k] = Path(fp[0].replace('Raw images', 'used for classification generated [aftermask_subbk200]').replace('.tif', '_aftermask_subbk.tif').replace('/mnt/smb_share/SML/Aging_classifier/Data_250421_multi-classification/Training set/', '/mnt/d/Data/SML/Aging_classification_dataset/mito_0421/'))
                if '-' in sheet:
                    sheet = sheet.split(' - ')[0]
                    
                dict_multi_classes.setdefault(str(sheet), []).extend(file_paths)
                
                class_num_list.append(len(file_paths))
                
        return dict_multi_classes, class_num_list

    def extract_train_dict(self, dict_, num, balanced = False, num_constant = 0):
        if isinstance(num, float):
            assert num < 1.
        if balanced:
            num_min = 1000
            for k, item_list in dict_.items():
                num_ = len(item_list)
                if num_ < num_min:
                    num_min = num_
            num = num_min - 48 # test minimum number : 24
        if balanced and (num_constant > 0):
            num_min = min([len(dict_[k]) for k in dict_.keys()])
            num = min(num_constant, num_min - 20)
        extracted_dict = {}
        for k, item_list in dict_.items():
            extracted_dict.update({k: item_list[: num if isinstance(num, int) else max(1, int(num * len(item_list)))]})
        return extracted_dict


    def extract_test_dict(self, dict_, train_dict, whether_eval = False, val_num = 0, excel_ = False):
        if not excel_:
            extract_eval_dict = {}
            extract_test_dict = {}
            for k, item_list in dict_.items():
                assert k in train_dict.keys()
                test_list = set(item_list).difference(set(train_dict[k]))
                if val_num > 0:
                    num_eval = val_num
                else:
                    num_eval = len(test_list) // 2
                if whether_eval:
                    extract_eval_dict.update({k: list(test_list)[:num_eval] if len(test_list) > 0 else []})
                    extract_test_dict.update({k: list(test_list)[num_eval:] if len(test_list) > 0 else []})
                else:
                    extract_test_dict.update({k: test_list if len(test_list) > 0 else []})
                    extract_eval_dict = extract_test_dict
            
        else:
            self.logger.info('reading target excel to get test data path')
            extract_eval_dict = {}
            extract_test_dict = {}
            excel_path = self.hp.data.excel_path_val_test
            import csv
            for p in Path(excel_path).iterdir():
                if p.name.startswith('.'):
                    continue
                for k, item_list in dict_.items():
                    if k.strip() in p.name:
                        path_root = Path(*item_list[0].parts[:-3])
                        with open(p, 'r') as csv_f:
                            name_list = [row[0] for row in csv.reader(csv_f)][:-5] # [:-5] [:-2]
                        extract_test_dict.update({k: [path_root / n for n in name_list]})
            extract_eval_dict = extract_test_dict

        return extract_eval_dict, extract_test_dict

    def print_data(self, train_dict, eval_dict, test_dict, save_excel, excel_save_path, save_path_tail):
        excel_save_path = Path(excel_save_path) / save_path_tail
        excel_save_path.mkdir(exist_ok = True, parents = True)

        assert list(train_dict.keys()) == list(test_dict.keys()) == list(eval_dict.keys()), \
            Exception('Train: {}, Eval: {}, Test: {}'.format(list(train_dict.keys()), list(eval_dict.keys()), list(test_dict.keys())))
        if self.logger is not None:
            print_head = 'self.logger.info'
        else:
            print_head = 'print'

        eval(print_head)('^^^ Train & Eval & Test data ^^^')
        eval(print_head)('Class_name: {}'.format(list(train_dict.keys())))
        if eval_dict is not None:
            eval(print_head)('Data_num: {}'.format([f'{len(train_dict[x])} / {len(eval_dict[x])} / {len(test_dict[x])}' for x in train_dict.keys()]))
        else:
            eval(print_head)('not use Evaluation data')
            eval(print_head)('Data_num: {}'.format([f'{len(train_dict[x])} / {len(test_dict[x])}' for x in train_dict.keys()]))
        
        if save_excel:
            self.save_excels(train_dict, excel_save_path, 'train_dataset_used.csv')
            self.save_excels(eval_dict, excel_save_path, 'val_dataset_used.csv')
            self.save_excels(test_dict, excel_save_path, 'test_dataset_used.csv')

    def dict2list(self, dict_list):
        if dict_list is not None:
            list_ = []
            for k, item_list in dict_list.items():
                for i in item_list:
                    list_.append({i: k})
            
            return list_
        else:
            return None
    
    def save_excels(self, dict_, save_path, path_name):
        import pandas as pd

        save_P = Path(save_path)
        save_P.mkdir(exist_ok = True, parents = True)

        with pd.ExcelWriter(save_P / path_name, engine = 'openpyxl') as writer:
            for sheet_name, values in dict_.items():
                df = pd.DataFrame(values, columns = ["Path"])
                df.to_excel(writer, sheet_name = sheet_name, index = False)

class DataAugment:
    def __init__(self):
        super().__init__()
    
    def random_blur(self):
        kernel_size = random.choice([3,5])
        return transforms.GaussianBlur(kernel_size) 
    
    def random_erase(self):
        return transforms.RandomErasing(
            p = 0.3,
            scale = (0.01, 0.1),
            ratio = (0.3, 0.33),
            value = 0,
            inplace = False
        )

    def __call__(self, img, type_ = None):
        assert type_ in ['blur', 'erase'], Exception(f"{type_} shouled be in [blue, erase]")
        if type_ == 'blur':
            return self.random_blur()(img)
        else:
            return self.random_erase()(img)
