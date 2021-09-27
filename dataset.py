import torch
import torchvision

from ptpt.log import error

from pathlib import Path

def get_dataset(task: str):
    if task in ['ffhq1024','ffhq1024-large']:
        transforms = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])
        dataset = torchvision.datasets.ImageFolder('data/ffhq1024', transform=transforms)
        train_idx, test_idx = torch.arange(0, 60_000), torch.arange(60_000, len(dataset))
        train_dataset, test_dataset = torch.utils.data.Subset(dataset, train_idx), torch.utils.data.Subset(dataset, test_idx)
    elif task == 'ffhq256':
        transforms = torchvision.transforms.Compose([
            torchvision.transforms.Resize(256),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])
        dataset = torchvision.datasets.ImageFolder('data/ffhq1024', transform=transforms)
        train_idx, test_idx = torch.arange(0, 60_000), torch.arange(60_000, len(dataset))
        train_dataset, test_dataset = torch.utils.data.Subset(dataset, train_idx), torch.utils.data.Subset(dataset, test_idx)
    elif task == 'ffhq128':
        transforms = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])
        dataset = torchvision.datasets.ImageFolder('data/ffhq128', transform=transforms)
        train_idx, test_idx = torch.arange(0, 60_000), torch.arange(60_000, len(dataset))
        train_dataset, test_dataset = torch.utils.data.Subset(dataset, train_idx), torch.utils.data.Subset(dataset, test_idx)
    elif task == 'cifar10':
        transforms = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])
        train_dataset = torchvision.datasets.CIFAR10('data', train=True, transform=transforms, download=True)
        test_dataset = torchvision.datasets.CIFAR10('data', train=False, transform=transforms, download=True)
    elif task == 'mnist':
        transforms = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
        ])
        train_dataset = torchvision.datasets.MNIST('data', train=True, transform=transforms, download=True)
        test_dataset = torchvision.datasets.MNIST('data', train=False, transform=transforms, download=True)
    elif task == 'kmnist':
        transforms = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
        ])
        train_dataset = torchvision.datasets.KMNIST('data', train=True, transform=transforms, download=True)
        test_dataset = torchvision.datasets.KMNIST('data', train=False, transform=transforms, download=True)
    else:
        msg = f"unrecognised dataset '{task}'!"
        error(msg)
        raise ValueError(msg)

    return train_dataset, test_dataset
