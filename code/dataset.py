#!/usr/bin/env mdl
import my_snip.base as base
import numpy as np
from my_snip.imgproc import gaussian_noise
import pickle

class Dataset(base.BaseDataset):

    base_path = '../data/cifar-10-batches-py/data_batch_'
    imgs = []
    labels = []

    minibatch_size = 32
    def __init__(self, batch_size = 128, dataset_name = 'train'):
        super().__init__(dataset_name)

        assert dataset_name in ['train', 'val']
        self.minibatch_size = batch_size

        self.instance_per_epoch = 50000

        with open("../data/meanstd.data", "rb") as f:
            self.mean, self.std = pickle.load(f)
        if dataset_name != 'train':
            self.instance_per_epoch = 10000
            self.base_path = '../data/cifar-10-batches-py/test_batch'

    def load(self):
        def unpickle(file):
            import pickle
            with open(file, 'rb') as fo:
                dict = pickle.load(fo, encoding='bytes')
            print(file, ' Done reading')
            return dict
        if self.dataset_name != 'train':
            t_dict = unpickle(self.base_path)
            #img = t_dict.values[0]
            self.imgs = t_dict[b'data'].reshape(10000, 32, 32, 3)
            self.imgs = np.transpose(self.imgs, (0, 3, 1, 2))
            self.imgs = self.imgs.astype(np.float32)
            #self.imgs = (self.imgs - self.mean) / self.std
            self.labels = np.array(t_dict[b'labels'])
            self.labels = self.labels.astype(np.int32)

            print('Loading Done! Instance_per_epoch: {}'.format(self.instance_per_epoch))
            return
        for i in range(1, 6):
            img_path = self.base_path + str(i)

            t_dict = unpickle(img_path)
            img = t_dict[b'data'].reshape(10000, 32, 32, 3)
            img = np.transpose(img, (0, 3, 1, 2))
            img = img.astype(np.float32)
            #img = (img - self.mean) / self.std

            label = np.array(t_dict[b'labels'])
            label = label.astype(np.int32)
            print(np.array(label).shape)
            #self.labels.append(label)
            if i == 1:
                self.imgs = img
                self.labels = label
            else:
                self.imgs = np.concatenate((self.imgs, img), axis = 0)
                self.labels = np.concatenate((self.labels, label), axis = 0)

        print('Loading Done! Instance_per_epoch: {}'.format(self.instance_per_epoch))
    def augument(self, img):
        img = gaussian_noise(img, sigma = 2)
        return img

    def instance_generator(self, encoded = False):
        indexes = np.arange(self.instance_per_epoch)

        while True:
            np.random.shuffle(indexes)
            for i in indexes:
                img, label = self.imgs[i], self.labels[i]
                img = img.astype(np.float32)#.transpose(2, 0, 1)
                label = label.astype(np.int32)

                #if self.dataset_name == 'train':
                    #img = self.augument(img)


                yield {
                    'data': img,
                   # 'data': img[np.newaxis],
                    'label': label
                }



def test():
    ds_train = Dataset(dataset_name='train')

    ds_train.load()



    sample = next(ds_train.instance_generator())

    data = sample['data']

    #print(data.shape)

    #print(sample['label'].shape)

    print(ds_train.mean)
    print(ds_train.std)

    print(ds_train.mean.shape)
    print(ds_train.std.shape)

if __name__ == '__main__':
    test()
# vim: ts=4 sw=4 sts=4 expandtab
