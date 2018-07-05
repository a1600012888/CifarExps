import torch
from tqdm import tqdm
from my_snip.clock import AvgMeter
import time
from my_snip.metrics import torch_accuracy
from collections import OrderedDict

def adversairal_train_one_epoch(net, optimizer, batch_generator, criterion, AttackMethod, clock, attack_freq = 1, use_adv = True):
    """
    adversarial training.

    :param net:
    :param optimizer:
    :param batch_generator:
    :param criterion:
    :param AttackMethod:  the attack method
    :param clock: clock object from my_snip.clock import TrainClock
    :param attack_freq:  Frequencies of training with adversarial examples
    :return:
    """

    training_losses = AvgMeter()
    training_accs = AvgMeter()

    clean_losses = AvgMeter()
    clean_accs = AvgMeter()

    defense_losses = AvgMeter()
    defense_accs = AvgMeter()
    names = ['loss', 'acc', 'clean_loss', 'clean_acc', 'adv_loss', 'adv_acc']

    clean_batch_times = AvgMeter()
    ad_batch_times = AvgMeter()
    net.train()
    clock.tock()

    pbar = tqdm(batch_generator)

    start_time = time.time()
    for mn_batch in pbar:
        clock.tick()

        data = torch.tensor(mn_batch['data'], dtype=torch.float32).cuda()
        label = torch.tensor(mn_batch['label'], dtype=torch.int64).cuda()


        data_time = time.time() - start_time

        if clock.minibatch % (attack_freq + 1) == 1 and use_adv:

            adv_inp = AttackMethod.attack(net, data, label)

            net.train()
            optimizer.zero_grad()

            pred = net(adv_inp)
            loss = criterion(pred, label)

            loss.backward()

            optimizer.step()

            defense_losses.update(loss.item())

            acc = torch_accuracy(pred, label, (1, ))

            defense_accs.update(acc[0].item())

            batch_time = time.time() - start_time
            ad_batch_times.update(batch_time)
        else:
            optimizer.zero_grad()
            pred = net(data)
            loss = criterion(pred, label)
            loss.backward()
            optimizer.step()
            acc = torch_accuracy(pred, label, (1, ))

            clean_losses.update(loss.item())
            clean_accs.update(acc[0].item())

            batch_time = time.time() - start_time
            clean_batch_times.update(batch_time)

        training_losses.update(loss.item())
        training_accs.update(acc[0].item())

        pbar.set_description("Training Epoch: {}".format(clock.epoch))

        values = [training_losses.mean, training_accs.mean, clean_losses.mean, clean_accs.mean, defense_losses.mean,
                  defense_accs.mean]
        pbar_dic = OrderedDict()
        for n, v in zip(names, values):
            pbar_dic[n] = v
        pbar_dic['clean_time'] = "{:.1f}".format(clean_batch_times.mean)
        pbar_dic['ad_time'] = "{:.1f}".format(ad_batch_times.mean)
        pbar.set_postfix(pbar_dic)
        '''
        pbar.set_postfix(
            loss = '{:.2f}'.format(training_losses.mean),
            acc = '{:.2f}'.format(training_accs.mean),
            clean_losses = '{:.2f}'.format(clean_losses.mean),
            clean_acc = "{:.2f}".format(clean_accs.mean),
            defense_losses = "{:.2f}".format(defense_losses.mean),
            defense_accs = "{:.2f}".format(defense_accs.mean),
            clean_time = "{:.2f}".format(clean_batch_times.mean),
            ad_time = "{:.2f}".format(ad_batch_times.mean)
        )'''

        start_time = time.time()

    #names = ['loss', 'acc', 'clean_loss', 'clean_acc', 'adv_loss', 'adv_acc']
    values = [training_losses.mean, training_accs.mean, clean_losses.mean, clean_accs.mean, defense_losses.mean,
              defense_accs.mean]

    dic = {n:v for n,v in zip(names, values)}
    dic = OrderedDict(dic)
    return dic


def adversarial_val(net, batch_generator, criterion, AttackMethod, clock, attack_freq = 1):
    """
        val both on clean data and adversarial examples.

        :param net:
        :param batch_generator:
        :param criterion:
        :param AttackMethod:  the attack method
        :param clock: clock object from my_snip.clock import TrainClock
        :param attack_freq:  Frequencies of training with adversarial examples
        :return:
        """

    training_losses = AvgMeter()
    training_accs = AvgMeter()

    clean_losses = AvgMeter()
    clean_accs = AvgMeter()

    defense_losses = AvgMeter()
    defense_accs = AvgMeter()
    names = ['loss', 'acc', 'clean_loss', 'clean_acc', 'adv_loss', 'adv_acc']

    clean_batch_times = AvgMeter()
    ad_batch_times = AvgMeter()
    net.eval()


    pbar = tqdm(batch_generator)

    start_time = time.time()
    i = 0
    for mn_batch in pbar:

        i += 1
        data = torch.tensor(mn_batch['data'], dtype=torch.float32).cuda()
        label = torch.tensor(mn_batch['label'], dtype = torch.int64).cuda()


        data_time = time.time() - start_time

        if i % (attack_freq + 1) == 1:

            adv_inp = AttackMethod.attack(net, data, label)

            net.eval()

            with torch.no_grad():
                pred = net(adv_inp)
                loss = criterion(pred, label)

                defense_losses.update(loss.item())

                acc = torch_accuracy(pred, label, (1,))

                defense_accs.update(acc[0].item())

                batch_time = time.time() - start_time
                ad_batch_times.update(batch_time)
        else:

            with torch.no_grad():
                pred = net(data)
                loss = criterion(pred, label)
                acc = torch_accuracy(pred, label, (1,))

                clean_losses.update(loss.item())
                clean_accs.update(acc[0].item())

                batch_time = time.time() - start_time
                clean_batch_times.update(batch_time)

        training_losses.update(loss.item())
        training_accs.update(acc[0].item())

        pbar.set_description("Validation Epoch: {}".format(clock.epoch))

        values = [training_losses.mean, training_accs.mean, clean_losses.mean, clean_accs.mean, defense_losses.mean,
                  defense_accs.mean]
        pbar_dic = OrderedDict()
        for n, v in zip(names, values):
            pbar_dic[n] = v
        pbar.set_postfix(pbar_dic)
        pbar_dic['clean_time'] = "{:.2f}".format(clean_batch_times.mean)
        pbar_dic['ad_time'] = "{:.2f}".format(ad_batch_times.mean)
        '''
        pbar.set_postfix(
            loss='{:.2f}'.format(training_losses.mean),
            acc='{:.2f}'.format(training_accs.mean),
            clean_losses='{:.2f}'.format(clean_losses.mean),
            clean_acc="{:.2f}".format(clean_accs.mean),
            defense_losses="{:.2f}".format(defense_losses.mean),
            defense_accs="{:.2f}".format(defense_accs.mean),
            clean_time="{:.2f}".format(clean_batch_times.mean),
            ad_time="{:.2f}".format(ad_batch_times.mean)
        )
        '''
        start_time = time.time()


    values = [training_losses.mean, training_accs.mean, clean_losses.mean, clean_accs.mean, defense_losses.mean,
              defense_accs.mean]

    dic = {n:v for n, v in zip(names, values)}
    dic = OrderedDict(dic)
    return dic