import argparse
import yaml
import itertools
import traceback
import random
import os
import torch
import torch.nn as nn
import torchvision
import torch.distributed as dist
import torch.multiprocessing as mp


def setup(hp_net, rank, world_size):
    os.environ["MASTER_ADDR"] = hp_net.train.dist.master_addr
    os.environ["MASTER_PORT"] = hp_net.train.dist.master_port

    # initialize the process group
    dist.init_process_group(backend="nccl" if dist.is_nccl_available() else "gloo",
                            rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def distributed_run(fn,  hp_net, world_size, vis = None):
    mp.spawn(fn, args=(hp_net, world_size, vis), nprocs=world_size, join=True)


def train_loop(rank, hp_net, world_size=0):
    if hp_net.model.device == "cuda" and world_size != 0:
        hp_net.model.device = rank
        setup(hp_net, rank, world_size)
        torch.cuda.set_device(hp_net.model.device)
    
    set_random_seed(hp_net.train.random_seed)

    # setup logger / writer
    if rank != 0:
        logger = None
        writer = None
    else:
        # set logger
        logger = make_logger(hp_net, hp_fpm = None)
        hp_str = yaml.dump(hp_net.to_dict())
        if hp_net.log.print_yaml:
            logger.info("Config:")
            logger.info(hp_str)
        if hp_net.data.train_dir_mit == "" or hp_net.data.train_dir_nuc == "":
            logger.error("train data directory cannot be empty.")
            raise Exception("Please specify directories of data")
        logger.info("Set up train process")

    # Sync dist processes (because of download MNIST Dataset), 同步所有进程，使其一致，防止其中一个在训练一个在测试
    if hp_net.model.device == "cuda" and world_size != 0:
        dist.barrier()

    # init Model
    channels_init_dict = {
        'temporal': 2,
        '3D-WF (continuous)': 10,
        '3D-WF (triple)': 3,
        '3D-WF (bi-triple)': 6,
        '3D-WF (triple2X)': 6 # NOTE: 暂时为12 应该修改为6 
    }
    
    if hp_net.data.organelle not in ['mito + nucleus', 'mask', 'nucleus']:
        if ' -t' in hp_net.data.mode:
           eval(f'hp_net.model.{hp_net.model.name}').channels = channels_init_dict['temporal']
        else:
            try:
                eval(f'hp_net.model.{hp_net.model.name}').channels = channels_init_dict[hp_net.data.mode]
            except:
                eval(f'hp_net.model.{hp_net.model.name}').channels = 1

    elif hp_net.data.organelle in ['mask', 'nucleus']:
        eval(f'hp_net.model.{hp_net.model.name}').channels = 1
    else:
        eval(f'hp_net.model.{hp_net.model.name}').channels = eval(f'hp_net.model.{hp_net.model.name}').channels # 7 12 


    if hp_net.data.img_edge == 'sobel':
        eval(f'hp_net.model.{hp_net.model.name}').channels = eval(f'hp_net.model.{hp_net.model.name}').channels * 2  
    elif hp_net.data.img_edge == 'wavelet':
        eval(f'hp_net.model.{hp_net.model.name}').channels = eval(f'hp_net.model.{hp_net.model.name}').channels * 4
        if hp_net.model.name != 'DenseNet121':
            eval(f'hp_net.model.{hp_net.model.name}').image_size = 512
    else: 
        pass

    # multi classes num 
    # eval(f'hp_net.model.{hp_net.model.name}').num_classes = len(hp_net.data.class_list)
    eval(f'hp_net.model.{hp_net.model.name}').num_classes = 5

    net_arch = Net_arch(hp_net)
    loss_f = loss_used(hp_net)
    model = Model(hp_net, net_arch, loss_f, rank, world_size)

    # load training state / network checkpoint
    if hp_net.load_multi.resume_state_path is not None:
        model.load_training_state(logger, ckpt = hp_net.load_multi.resume_state_path)
    elif hp_net.load_multi.network_chkpt_path is not None:
        model.load_network(logger=logger, ckpt = hp_net.load_multi.network_chkpt_path)
    else:
        if logger is not None:
            logger.info("Starting new training run.")

    try:
        if world_size == 0 or hp_net.data.divide_dataset_per_gpu:
            epoch_step = 1
        else:
            epoch_step = world_size
        acc_best = 0.0
        eval_acc = 0.0
        best_idx = 1
        if not hp_net.log.generate:
            dataloader_ = create_dataloader_AL(hp_net, rank, world_size, logger, multi_classes = True)
            # make dataloader
            if logger is not None:
                logger.info("Making train dataloader...")
            train_loader = dataloader_.get_dataloader(DataloaderMode.train)
            if logger is not None:
                logger.info("Making eval dataloader...")
            eval_loader = dataloader_.get_dataloader(DataloaderMode.eval)
            if logger is not None:
                logger.info("Making test dataloader...")
            test_loader = dataloader_.get_dataloader(DataloaderMode.test)
            
            # logger.info("Testing before training")
            # test_model(hp_net, model, test_loader = test_loader[0], writer = writer, grad_CAM = False, logger = logger)

            if not hp_net.train.Active_Learning.switch_on:
                with tqdm(total = hp_net.train.num_epoch) as t:
                    t.set_description('training ---> ')
                    for model.epoch in itertools.count(model.epoch + 1, epoch_step):
                        # set_random_seed(420)
                        if model.epoch > hp_net.train.num_epoch or hp_net.log.generate: # set 200 epochs
                            break
                        train_model(hp_net, model, train_loader[0], writer, logger)
                        model.run_lr_scheduler()
                        if hp_net.log.val_interval and ((model.epoch + 1) % hp_net.log.val_interval == 0):
                            eval_acc = eval_model(hp_net, model, eval_loader = eval_loader[0], writer = writer, logger = logger)
                        # if model.epoch == hp_net.train.num_epoch or (hp_net.log.chkpt_interval and model.epoch + 1 % hp_net.log.chkpt_interval == 0):
                        if (model.epoch == hp_net.train.num_epoch) or ((hp_net.log.chkpt_interval > 0) and (model.epoch + 1 % hp_net.log.chkpt_interval == 0)):
                            # model.save_network(logger)
                            model.save_training_state(logger, best_ = False)
                        if (model.epoch >= 10) and ((model.epoch + 1) %  hp_net.log.val_interval == 0) and (eval_acc >= acc_best): 
                            model.save_training_state(logger, best_ = True, best_idx = best_idx)
                            acc_best = eval_acc
                            best_idx += 1
                        t.update(n = 1)
            else:
                logger.info('<<< Active Learning is on >>>')
                with tqdm(total = hp_net.train.Active_Learning.rounds) as t:
                    t.set_description('Active Learning ---> ')
                    for round in range(hp_net.train.Active_Learning.rounds):
                        logger.info(f'Round: {round}')
                        ## TODO
                        # if round < 1:
                        #     pass
                        # else:
                        #     uncertainty = model.get_uncertainty(train_loader[1])
                        #     indices_to_label = torch.argsort(torch.tensor(uncertainty), descending = True)[]
                        #
                        #





            if logger is not None:
                logger.info("End of Train")
            if hp_net.log.show_last:
                logger.info("Testing")
                test_model(hp_net, model, test_loader = test_loader[0], writer = writer, grad_CAM = False, logger = logger)
        else:
            assert hp_net.load_multi.resume_state_path is not None, Exception('state_path should not be None when generating mode')

            if hp_net.data.dual_class == False:
                logger.info('Generating Multi_classes')
                model_generate_Multi(hp_net, model)
            else:
                logger.info('Generating Dual_classes')
                model_generate_dual(hp_net, model)


    except Exception as e:
        if logger is not None:
            logger.error(traceback.format_exc())
        else:
            traceback.print_exc()
    finally:
        if hp_net.model.device == "cuda" and world_size != 0:
            cleanup()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        # "-Net_c", "--Net_config", type=str, default = r'config/Net_AC_mito.yaml', # nuc, mito
        # "-Net_c", "--Net_config", type=str, default = r'config/Net_AC_nuc.yaml', # nuc, mito
        # "-Net_c", "--Net_config", type=str, default = r'config/Net_AC_nuc_2_class.yaml', # nuc, mito
        # "-Net_c", "--Net_config", type=str, default = r'config/Net_AC_mito_2_class.yaml', # nuc, mito
        # "-Net_c", "--Net_config", type=str, default = r'config/Net_AC_mito_No6.yaml', # nuc, mito
        # "-Net_c", "--Net_config", type=str, default = r'config/Net_AC_mito_No2.yaml', # nuc, mito
        # "-Net_c", "--Net_config", type=str, default = r'config/Net_AC_mito_No1.yaml', # nuc, mito
        # "-Net_c", "--Net_config", type=str, default = r'config/Net_AC_mito_No5.yaml', # nuc, mito
        # "-Net_c", "--Net_config", type=str, default = r'config/Net_AC_nuc_No6.yaml', # nuc, mito
        # "-Net_c", "--Net_config", type=str, default = r'config/Net_AC_mito_No4_RE_new_data.yaml', # nuc, mito
        "-Net_c", "--Net_config", type=str, default = r'config/Net_AC_mito_No6_RE_new_data.yaml', # nuc, mito
        # "-Net_c", "--Net_config", type=str, default = r'config/Net_AC_mito_No5_RE_new_data.yaml', # nuc, mito
        # "-Net_c", "--Net_config", type=str, default = r'config/Net_AC_mito_No3_RE_new_data.yaml', # nuc, mito
        # "-Net_c", "--Net_config", type=str, default = r'config/Net_AC_mito_No6_RE_small.yaml', # nuc, mito
        # "-Net_c", "--Net_config", type=str, default = r'config/Net_AC_mito_No4_RE.yaml', # nuc, mito
        help="Net yaml file for config."
    )
    args = parser.parse_args()

    hp_net = load_hparam(args.Net_config)

    hp_net.model.device = hp_net.model.device.lower()
    hp_net.yaml_dir = args.Net_config


    # random seed
    if hp_net.train.random_seed is None:
        hp_net.train.random_seed = random.randint(1, 10000)

    set_random_seed(hp_net.train.random_seed)

    # # set GPUs used
    # os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    # os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, hp_net.train.dist.gpus))

    if hp_net.train.dist.gpus[0] < 0:
        hp_net.train.dist.gpus = [torch.cuda.device_count()]

    if hp_net.model.device == "cpu" or hp_net.train.dist.gpus == [0]:
        train_loop(0, hp_net)
    else:
        assert torch.cuda.is_available() and torch.cuda.device_count() >= len(hp_net.train.dist.gpus), \
            'cuda.is_available: {}, cuda.device_count: {}'.format(torch.cuda.is_available(), torch.cuda.device_count())
        distributed_run(train_loop,hp_net, len(hp_net.train.dist.gpus))


if __name__ == "__main__":

    from util.util import load_hparam, set_random_seed
    
    from model.model_arch import Net_arch
    from model.model_AC import Model
    from util.train_model import train_model
    from util.test_model import test_model
    from util.eval_model import eval_model
    from util.model_apply import model_generate as model_generate_Multi
    from util.model_apply_dual import model_generate as model_generate_dual
    from util.logger import make_logger
    from loss.loss import loss_used
    from dataset.dataloader_AC_ActivateLearning import create_dataloader_AL, DataloaderMode
    from tqdm import tqdm

    set_random_seed(420)
    main()
