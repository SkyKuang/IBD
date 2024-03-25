from __future__ import print_function
import torch
import torchvision
from torchvision import datasets
import torchvision.transforms as transforms
import torch.utils.data as Data
import numpy as np
from PIL import Image


def get_transform(img_size=32):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4, padding_mode='edge'),
        transforms.RandomHorizontalFlip(),
        transforms.Resize(img_size),
        transforms.ToTensor(),
    ])

    transform_test = transforms.Compose([
        # transforms.Resize(img_size),
        transforms.ToTensor(),
    ])
    
    return transform_train, transform_test


def get_transform_svhn(img_size=32):
    transform_train = transforms.Compose([
        transforms.ToTensor(),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])

    return transform_train, transform_test

def data_loader(dataset='cifar10', root=None, train_batch=128, test_batch=100, transform=None):
    if dataset == 'cifar10':
        if root is None:
            root = './../../datasets/cifar10'
        if transform is None:
            transform = get_transform(img_size=32)

        # trainset = torchvision.datasets.CIFAR10(root=root,
        trainset = CIFAR10(root=root,
                                                train=True,
                                                download=True,
                                                transform=transform[0])
        testset = torchvision.datasets.CIFAR10(root=root,
                                            train=False,
                                            download=True,
                                            transform=transform[1])
        classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    elif dataset == 'cifar100':
        if root is None:
            root = './../../datasets/cifar100'
        if transform is None:
            transform = get_transform(img_size=32)

        # trainset = torchvision.datasets.CIFAR100(root=root,
        trainset = CIFAR100(root=root,
                                                train=True,
                                                download=True,
                                                transform=transform[0])
        testset = torchvision.datasets.CIFAR100(root=root,
                                            train=False,
                                            download=True,
                                            transform=transform[1])

    elif dataset == 'svhn':
        if root is None:
            root = './../../datasets/svhn'
        if transform is None:
            transform = get_transform_svhn(img_size=32)
        trainset = SVHN(root=root,
                                                split='train',
                                                download=True,
                                                transform=transform[0])
        testset = torchvision.datasets.SVHN(root=root,
                                                split='test',
                                                download=True,
                                                transform=transform[1])

    elif dataset == 'mnist':
        if root is None:
            root = './../../datasets/mnist'
        if transform is None:
            transform = get_transform(img_size=28)

        trainset = torchvision.datasets.MNIST(root=root,
                                                train=True,
                                                download=True,
                                                transform=transform[0])
        testset = torchvision.datasets.MNIST(root=root,
                                                train=False,
                                                download=True,
                                                transform=transform[1])

    elif dataset == 'tiny':
        from .crd_tinyImage import load_TinyImageNet
        trainset, testset = load_TinyImageNet(batch_size=train_batch,
                                                test_size=test_batch,
                                                size=64, resize=64)

    else:
        pass

    trainloader = torch.utils.data.DataLoader(trainset,
                                            batch_size=train_batch,
                                            shuffle=True,
                                            num_workers=2)

    testloader = torch.utils.data.DataLoader(testset,
                                            batch_size=test_batch,
                                            shuffle=True,
                                            num_workers=2)
    return trainloader, testloader


def crd_data_loader(dataset='cifar10', batch_size_train=128, crd_k=1, crd_m=1, crd_mode='exact'):
    if dataset == 'cifar10':
        from .crd_cifar10 import get_cifar10_dataloaders_sample
        trainloader, testloader, n_data = get_cifar10_dataloaders_sample(batch_size=batch_size_train,
                                                                        num_workers=2, k=crd_k, m=crd_m, mode=crd_mode)

    elif dataset == 'cifar100':
        from .crd_cifar100 import get_cifar100_dataloaders_sample
        trainloader, testloader, n_data = get_cifar100_dataloaders_sample(batch_size=batch_size_train,
                                                                        num_workers=2, k=crd_k, m=crd_m, mode=crd_mode)

    elif dataset == 'svhn':
        from .crd_svhn import get_svhn_dataloaders_sample
        trainloader, testloader, n_data = get_svhn_dataloaders_sample(batch_size=batch_size_train,
                                                                        num_workers=0, k=crd_k, m=crd_m, mode=crd_mode)
    elif dataset == 'mnist':
        from .crd_mnist import get_mnist_dataloaders_sample
        trainloader, testloader, n_data = get_mnist_dataloaders_sample(batch_size=batch_size_train,
                                                                        num_workers=0, k=crd_k, m=crd_m, mode=crd_mode)
    elif args.dataset == 'tiny':
        from .crd_tinyImage import get_tiny_dataloaders_sample
        trainloader, testloader, n_data = get_tiny_dataloaders_sample(batch_size=batch_size_train,
                                                                        num_workers=0, k=crd_k, m=crd_m, mode=crd_mode)
    return trainloader, testloader, n_data

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def get_data(train=False):
    data = None
    labels = None
    data_dir = './../../datasets/cifar10/'
    if train:
        for i in range(1, 6):
            batch = unpickle(data_dir + 'cifar-10-batches-py/data_batch_' + str(i))
            if i == 1:
                data = batch[b'data']
            else:
                data = np.concatenate([data, batch[b'data']])
            if i == 1:
                labels = batch[b'labels']
            else:
                labels = np.concatenate([labels, batch[b'labels']])

        data_tmp = data
        labels_tmp = labels
        # repeat n times for different masks
        mask_num = 10
        for i in range(mask_num - 1):
            data = np.concatenate([data, data_tmp])
            labels = np.concatenate([labels, labels_tmp])
    else:
        batch = unpickle(data_dir + 'cifar-10-batches-py/test_batch')
        data = batch[b'data']
        labels = batch[b'labels']
    return data, labels

def target_transform(label):
    label = np.array(label)
    target = torch.from_numpy(label).long()
    return target

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
])

class CIFAR10_Dataset(Data.Dataset):

    def __init__(self, train=True, target_transform=None):
        self.target_transform = target_transform
        self.train = train

        if self.train:
            self.train_data, self.train_labels = get_data(train)
            self.train_data = self.train_data.reshape((self.train_data.shape[0], 3, 32, 32))
            self.train_data = self.train_data.transpose((0, 2, 3, 1))
        else:
            self.test_data, self.test_labels = get_data()
            self.test_data = self.test_data.reshape((self.test_data.shape[0], 3, 32, 32))
            self.test_data = self.test_data.transpose((0, 2, 3, 1))

    def __getitem__(self, index):
        if self.train:
            img, label = self.train_data[index], self.train_labels[index]
        else:
            img, label = self.test_data[index], self.test_labels[index]

        img = Image.fromarray(img)
        if self.train:
            img = transform_train(img)
        else:
            img = transform_test(img)

        if self.target_transform is not None:
            target = self.target_transform(label)

        return img, target

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)

def get_ME_dataloader(train_batch=128, test_batch=100):

    train_dataset = CIFAR10_Dataset(True, target_transform)
    test_dataset = CIFAR10_Dataset(False, target_transform)

    if torch.cuda.is_available():
        n_gpu = torch.cuda.device_count()

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=train_batch,
                                               shuffle=True,
                                               num_workers=2)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=test_batch,
                                              shuffle=False,
                                              num_workers=2)
    return train_loader, test_loader


class CIFAR10(datasets.CIFAR10):
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
        super(CIFAR10, self).__init__(root, train=train, transform=transform,
                                     target_transform=target_transform, download=download)

        # unify the interface
        if not hasattr(self, 'data'):       # torch <= 0.4.1
            if self.train:
                self.data, self.targets = self.train_data, self.train_labels
            else:
                self.data, self.targets = self.test_data, self.test_labels

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index
    
    @property
    def num_classes(self):
        return 10


class CIFAR100(datasets.CIFAR100):
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
        super(CIFAR100, self).__init__(root, train=train, transform=transform,
                                     target_transform=target_transform, download=download)

        # unify the interface
        if not hasattr(self, 'data'):       # torch <= 0.4.1
            if self.train:
                self.data, self.targets = self.train_data, self.train_labels
            else:
                self.data, self.targets = self.test_data, self.test_labels

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index
    
    @property
    def num_classes(self):
        return 100


class SVHN(datasets.SVHN):
    def __init__(self, root, split='train',
                 transform=None, target_transform=None,download=False):
        super().__init__(root=root, split=split, download=download,
                         transform=transform, target_transform=target_transform)

        num_classes = 10
        if self.split == 'train':
            num_samples = len(self.data)
            label = self.labels
        else:
            num_samples = len(self.data)
            label = self.labels

    def __getitem__(self, index):
        if self.split == 'train':
            img, target = self.data[index], self.labels[index]
        else:
            img, target = self.data[index], self.labels[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = img.transpose(1,2,0)
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index
      