import torch
import torch.nn.functional as F
from tqdm import tqdm
# from mayavi import mlab
from pathlib2 import Path
import tifffile as tiff
import numpy as np
from torchvision import transforms
import kornia
import pywt

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

label2class = {
    0: 'Ctrl', 
    1: 'Doxo', 
    2: 'Eto', 
    3: 'Anti', 
    4: 'H2O2'
    }

class2label = {v: k for k, v in label2class.items()}


def get_path_list_Old_Young(path, key_name, dual_class = True):
    list_old = []
    list_young = []
    if path:
        for p in Path(path).iterdir():
            if p.is_file() or key_name not in p.name:
                continue
            # label = str(p).split('(')[-1].split('d')[0]
            name = str(p.name).split('-')[0]
            if name.strip().lower() != 'ctrl':  # and 'manually selected' in p.name:
                if dual_class:
                    label = '1.0'
                else:
                    label = class2label[name.strip()]
            elif name.strip().lower() == 'ctrl':
                label = '0'
            else:
                continue

            if label != '0':
                path_load(p, list_old, label)
            else:
                path_load(p, list_young, label)

        return list_old, list_young
    else:
        return None, None

def path_load_mask(path, p_list, label):
    try:
        for p_ in path.iterdir():
            if p_.suffix == '.tif' and 'generated_mask' in p_.parent.name:
                p_list.append([p_, label])
            elif p_.is_dir() and p_.name != 'compare':
                path_load_mask(p_, p_list, label)
            else:
                continue
    except:
        raise Exception()

def path_load(path, p_list, label):
    try:
        for p_ in path.iterdir():
            if p_.suffix == '.tif': 
                p_list.append([p_, label])
            elif p_.is_dir() and p_.name != 'compare' and p_.name != 'generated_mask':
                path_load(p_, p_list, label)
            else:
                pass
    except:
        raise Exception()
        pass

def sort_by_idx(list_):
    if list_ is not None:
        list_ = list(list_)
        return sorted(list_, key=lambda x: int(x[0].name.split('_')[0]))
    else:
        return None

def get_data_dict(young_list, old_list, test_list):
    if young_list is not None and old_list is not None:
        for p_ in old_list:
            test_list.append({p_[0]: p_[1]})
        for p_ in young_list:
            test_list.append({p_[0]: '0'})
        return test_list
    else:
        return test_list
    
def sorted_by_dir_name(list_dict, dir_groups = False, dir_name = 'all'):
    if dir_groups:
        dir_name_dict_list = {}
        for dict_ in list_dict:
            if (list(dict_.keys())[0].parts[-2] in dir_name) or dir_name == 'all': 
                p_save = Path(*list(dict_.keys())[0].parts[:-1])
                if p_save in dir_name_dict_list.keys():
                    dir_name_dict_list[p_save].append(dict_)
                else:
                    dir_name_dict_list.update({p_save: [dict_]})

    else:
        p_save = Path(*list(list_dict[0].keys())[0].parts[:-1])
        dir_name_dict_list = {p_save: list_dict}

    return dir_name_dict_list

def model_generate(hp, model, logger):
    model.net.eval()

    predict_probability_list = []

    for test_name in hp.data.test_name:

        inp_size = hp.data.img_size 
        AC_list_test_mit = []
        AC_list_old_test_mit, AC_list_young_test_mit = get_path_list_Old_Young(hp.data.test_dir_mit, key_name=test_name)
        AC_list_old_test_mit = sort_by_idx(AC_list_old_test_mit)
        AC_list_young_test_mit = sort_by_idx(AC_list_young_test_mit)

        get_data_dict(AC_list_young_test_mit, AC_list_old_test_mit, AC_list_test_mit)

        print(f'Testing {hp.data.organelle.upper()} in {test_name}')

        if hp.data.organelle == 'mito':
            AC_list_test_all = AC_list_test_mit
        else:
            raise ValueError(f"invalid organelle {hp.data.organelle}")
        
        AC_list_test_groups = sorted_by_dir_name(AC_list_test_all, hp.data.dir_group, hp.data.dir_name_list)

        for dir_name, AC_list_test in AC_list_test_groups.items():
            total_test_loss_state = {}
            total_acc = 0
            predict_probability_dict = {}
            predict_accuracy_dict = {}

            with torch.no_grad():
                for test_p_dict in tqdm(AC_list_test,
                                    total = len(AC_list_test), position = 0,
                                    leave = True, colour = 'BLUE', desc = 'Test'):
                    p_, label_ = list(test_p_dict.items())[0]
                    input_ = get_img(p_, [inp_size, inp_size], hp.data.img_edge, hp.data.organelle)

                    model.feed_data(input=input_, GT = torch.LongTensor([float(label_)]), GT_grad = None)
                    output, _, evaluate_loss_state, acc_evaluate = model.model_test(hp.data.dual_class)

                    if evaluate_loss_state is not None:
                        if list(total_test_loss_state.keys()) == []:
                            total_test_loss_state.update(evaluate_loss_state)
                        else:
                            for k, v in evaluate_loss_state.items():
                                total_test_loss_state[k] += v / len(AC_list_test)

                    if acc_evaluate is not None:
                        total_acc += acc_evaluate / len(AC_list_test)

                    output_prob = torch.nn.functional.softmax(output['output'], dim = 1)
                    predict_class = torch.argmax(output_prob)

                    if hp.data.dual_class:
                        if predict_class != 0:
                            predict_class = hp.data.sense_class
                        else:
                            predict_class = 'Ctrl'
                                      
                    for i, v in enumerate(model.GT):

                        predict_probability_list.append([str(p_), 
                                                         torch.nn.functional.softmax(output['output'], dim = 1).cpu().numpy()[0]])
                        
                        if str(float(v.item())) in predict_probability_dict.keys():
                            predict_probability_dict[str(float(v.item()))].append(
                                [str(p_), 
                                predict_class]
                                )
                            
                        else:
                            predict_probability_dict.update({str(float(v.item())): 
                                                            [[str(p_), 
                                                            predict_class]]}
                                                            )
    
                        if str(float(v.item())) in predict_accuracy_dict.keys():
                            predict_accuracy_dict[str(float(v.item()))][0] += acc_evaluate
                            predict_accuracy_dict[str(float(v.item()))][1] += 1
                        else:
                            predict_accuracy_dict.update({str(float(v.item())): [acc_evaluate, 1]})
                        
                        if logger is not None:
                            logger.info(f' Predict Class: {predict_class}, Ground Truth: {label2class[v.item()] if v.item() == 0 else label2class[2]}')
                        else:
                            pass

                if hp.log.save:

                    for k, v in predict_probability_dict.items():
                        print(f'Writing predict result for {test_name + " " + str(dir_name).split("/")[-1]}')
                        import csv
                        if hp.data.dir_group:
                            p_save_ = Path(*dir_name.parts[:-3]) / '{}_{}_Accuracy_{}_{:.4f}.csv'.format(dir_name.parts[-3], dir_name.parts[-1], hp.data.organelle, predict_accuracy_dict[k][0] / predict_accuracy_dict[k][1])
                        else:
                            p_save_ = Path(*dir_name.parts[:-2]) / '{}_Accuracy_{}_{:.4f}.csv'.format(dir_name.parts[-2], hp.data.organelle, predict_accuracy_dict[k][0] / predict_accuracy_dict[k][1])
                        print(p_save_)
                        with open(p_save_, 'w', newline = '') as csv_file:
                            writer = csv.writer(csv_file)
                            assert isinstance(v, list), Exception('v should be list here')
                            class_num = [0,0,0,0,0]
                            for v_ in v:
                                path_ = Path(v_[0])
                                path_save = str(Path(*path_.parts[-3:]))
                                row_ = [path_save, f'predict: {label2class[int(float(v_[1]))]}', f'GT: {"Young" if k == "0.0" else hp.data.sense_class}']
                                writer.writerow(row_)
                                class_num[int(float(v_[1]))] += 1
                            
                            for i in range(2):
                                ratio_ = class_num[i] / sum(class_num) * 100
                                writer.writerow([f'{label2class[i]}: {ratio_:.2f}%'])

                            csv_file.close()

    if logger is not None:
        logger.info('Test complete')     

def image_edge_sobel(img_tensor):
    img_sobel = kornia.filters.spatial_gradient(img_tensor.unsqueeze(0), mode='sobel', order=1, normalized=True).squeeze(0)
    return img_sobel

def image_edge_wavelet(img_tensor):
    img_wavelet_coeffs = pywt.dwt2(img_tensor, 'haar')
    LL, (LH, HL, HH) = img_wavelet_coeffs
    return torch.from_numpy(np.concatenate([LL, LH, HL, HH], axis = 0))

def get_img(p, img_size, img_edge, organelle):

    data_trans = transforms.Compose([transforms.Resize([img_size[0], img_size[0]])])
    img_ = tiff.imread(str(p))
    if img_.ndim < 3:
        img_ = img_[None, ...]
    img_ = torch.from_numpy(img_).type(torch.float32)
    img_ = data_trans(img_)
    if img_edge is not None:
        if img_edge == 'sobel':
            img_ = image_edge_sobel(img_)
        elif img_edge == 'wavelet':
            img_ = image_edge_wavelet(img_)
        else:
            raise ValueError()
    
    img_ = norm(img_)
    img_ = torch.clamp(img_, 0.0, 1.0)
#
    return img_.unsqueeze(0)

def norm(x):
    x = x - x.min()
    x = x / x.max()
    return x


