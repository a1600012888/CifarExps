import torch
from attack import IPGD
from my_snip.clock import AvgMeter
from tqdm import tqdm
import time

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

        for a in Attacks:
            defense_accs.update(a.get_batch_accuracy(net, data, label))

        pbar.set_description('Evulating Roboustness')

