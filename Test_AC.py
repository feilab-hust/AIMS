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

    # setup logger / writer
    if rank != 0:
        logger = None
    else:
        # set logger
        logger = make_logger(hp_net, hp_fpm = None)
        hp_str = yaml.dump(hp_net.to_dict())
        if hp_net.log.print_yaml:
            logger.info("Config:")
            logger.info(hp_str)
        logger.info("Set up Test process")

    # Sync dist processes (because of download MNIST Dataset), 同步所有进程，使其一致，防止其中一个在训练一个在测试
    if hp_net.model.device == "cuda" and world_size != 0:
        dist.barrier()

    if hp_net.data.img_edge == 'wavelet':
        eval(f'hp_net.model.{hp_net.model.name}').channels = eval(f'hp_net.model.{hp_net.model.name}').channels * 4
        eval(f'hp_net.model.{hp_net.model.name}').image_size = hp_net.train.image_size // 2

    net_arch = Net_arch(hp_net)
    loss_f = loss_used(hp_net)
    model = Model(hp_net, net_arch, loss_f, rank, world_size)

    try:
        # load training state / network checkpoint
        if hp_net.load.network_chkpt_path is not None:
            model.load_network(logger=logger, ckpt = hp_net.load.network_chkpt_path)
        else:
            logger.info("prertained ckpt is needed before testing")
            raise ValueError()

        if hp_net.data.dual_class == False:
            logger.info('Testing Multi Classification')
            model_generate_Multi(hp_net, model, logger)
        else:
            logger.info('Testing Dual Classification')
            model_generate_dual(hp_net, model, logger)

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
        # "-Net_c", "--Net_config", type=str, default = r'config/Net_AC_mito_MultiSeneClass.yaml',
        "-Net_c", "--Net_config", type=str, default = r'config/Net_AC_mito_Ctrl&Eto_U2OS.yaml',
        # "-Net_c", "--Net_config", type=str, default = r'config/Net_AC_mito_Ctrl&Anti_U2OS.yaml',
        # "-Net_c", "--Net_config", type=str, default = r'config/Net_AC_mito_Ctrl&Doxo_U2OS.yaml',
        # "-Net_c", "--Net_config", type=str, default = r'config/Net_AC_mito_Ctrl&OS_U2OS.yaml',
        # "-Net_c", "--Net_config", type=str, default = r'config/Net_AC_mito_Ctrl&Eto_crossCell.yaml',
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

    train_loop(0, hp_net)

if __name__ == "__main__":

    from util.util import load_hparam, set_random_seed
    
    from model.model_arch import Net_arch
    from model.model_AC import Model
    from util.model_apply import model_generate as model_generate_Multi
    from util.model_apply_dual import model_generate as model_generate_dual
    from util.logger import make_logger
    from loss.loss import loss_used

    main()
