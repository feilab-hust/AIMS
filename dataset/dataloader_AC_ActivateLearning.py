from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.data.distributed import DistributedSampler
from dataset.dataset_AC_multi import Dataset_AC_multi
from dataset.utils import DataloaderMode
from prefetch_generator import BackgroundGenerator

class DataLoader_PreG(DataLoader): 
    
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())
        

class create_dataloader_AL():
    def __init__(self, hp_net, rank, world_size, logger, multi_classes = False):
        super().__init__()
        self.dataset_train = Dataset_AC_multi(hp_net, DataloaderMode.train, logger)
        self.dataset_test = Dataset_AC_multi(hp_net, DataloaderMode.test, logger)
        self.train_use_shuffle = True
        self.sampler = None
        if world_size > 1 and hp_net.data.divide_dataset_per_gpu:
            self.sampler = DistributedSampler(self.dataset, world_size, rank)
            self.train_use_shuffle = False
        self.hp = hp_net
        self.logger = logger

    def init_dataset(self):
        init_ratio = self.hp.train.Activate_Learning.initial_data_ratio
        init_size = init_ratio * len(self.dataset)
        self.logger.info('Init dataset for Active Learning: init number: {}'.format(init_size))
        self.init_trainset, self.activate_learning_trainset = random_split(self.dataset, [init_size, len(self.dataset) - init_size])

    def update_dataset(self):
        pass

    def get_dataloader(self, mode):
        if mode is DataloaderMode.train:
            dataloader = DataLoader_PreG(
                dataset=self.dataset_train,
                batch_size=self.hp.train.batch_size,
                shuffle=self.train_use_shuffle,
                sampler=self.sampler,
                num_workers=self.hp.train.num_workers,
                pin_memory=True,
                drop_last=False,
            )
        elif mode is DataloaderMode.eval:
            dataloader = DataLoader_PreG(
                dataset=self.dataset_test,
                batch_size=self.hp.train.batch_size,
                shuffle=False,
                sampler=self.sampler,
                num_workers=self.hp.train.num_workers,
                pin_memory=True,
                drop_last=False,
            )
        elif mode is DataloaderMode.test:
            dataloader = DataLoader_PreG(
                dataset=self.dataset_test,
                batch_size=self.hp.test.batch_size,
                shuffle=False,
                sampler=self.sampler,
                num_workers=self.hp.test.num_workers,
                pin_memory=True,
                drop_last=False,
            )
        else:
            raise ValueError("invalid dataloader mode {}".format(self.mode))

        if self.hp.train.Active_Learning.switch_on and mode is DataloaderMode.train:
            active_learning_dataloader = DataLoader(
                dataset=self.activate_learning_trainset,
                batch_size=1,
                shuffle=False,
                sampler=self.sampler,
                num_workers=self.hp.train.num_workers,
                pin_memory=True,
                drop_last=False,
            )
            return [dataloader, active_learning_dataloader]
        else:
            return [dataloader]
