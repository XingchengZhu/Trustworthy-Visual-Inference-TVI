import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from .config import Config

def get_dataloaders():
    # Standard augmentation for training
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    # CIFAR-100 Mean/Std is different, technically we should adjust Normalize based on dataset
    # CIFAR-100: mean=(0.5071, 0.4867, 0.4408), std=(0.2675, 0.2565, 0.2761)
    if Config.DATASET_NAME == 'cifar100':
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])
        
        train_dataset = datasets.CIFAR100(root=Config.DATA_DIR, train=True, download=True, transform=train_transform)
        support_dataset = datasets.CIFAR100(root=Config.DATA_DIR, train=True, download=True, transform=test_transform)
        test_dataset = datasets.CIFAR100(root=Config.DATA_DIR, train=False, download=True, transform=test_transform)
        
    else: # Default CIFAR-10
        train_dataset = datasets.CIFAR10(root=Config.DATA_DIR, train=True, download=True, transform=train_transform)
        support_dataset = datasets.CIFAR10(root=Config.DATA_DIR, train=True, download=True, transform=test_transform)
        test_dataset = datasets.CIFAR10(root=Config.DATA_DIR, train=False, download=True, transform=test_transform)

    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=2)
    support_loader = DataLoader(support_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=2)

    return train_loader, support_loader, test_loader
