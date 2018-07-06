import torch
from attack import IPGD
from my_snip.clock import AvgMeter
from tqdm import tqdm
import time
import numpy as np
def evalRoboustness(net, batch_generator):

    defense_accs = AvgMeter()

    epsilons = [4, 8, 12, 16, 20, 24]
    nb_iters = [40, 80, 120]
    Attacks = []
    for e in epsilons:
        for nb in nb_iters:
            Attacks.append(IPGD(e, e//2, nb))

    net.eval()
    pbar = tqdm(batch_generator)

    for mn_batch in pbar:
        data = torch.tensor(mn_batch['data'], dtype=torch.float32).cuda()
        label = torch.tensor(mn_batch['label'], dtype=torch.int64).cuda()

        choices = np.random.randint(low = 0, high = 17, size = 4)
        for c in choices:
            defense_accs.update(Attacks[c].get_batch_accuracy(net, data, label))

        pbar.set_description('Evulating Roboustness')


def code_test():

    from base_model.cifar_resnet18 import cifar_resnet18
    from dataset import Dataset
    from my_snip.base import EpochDataset

    ds_val = Dataset(dataset_name = 'val')
    ds_val.load()
    ds_val = EpochDataset(ds_val)

    model_path = '../exps/exp0/checkpoint.pth.tar'

    net = cifar_resnet18()

    checkpoint = torch.load(model_path)

    net.load_state_dict(checkpoint['state_dict'])

    net.cuda()

    epoch = next(ds_val.epoch_generator())
    evalRoboustness(net, epoch)



if __name__ == '__main__':

    code_test()
