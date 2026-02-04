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
    support_loader = DataLoader(support_dataset, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=2)

    return train_loader, support_loader, test_loader

def get_ood_loader(ood_dataset_name):
    """
    Returns a dataloader for OOD evaluation.
    Transform must match the ID dataset's normalization roughly, or at least be tensor/resize.
    Ideally, we use the same normalization as ID (Backbone expects ID normalization).
    """
    # Standard ID transform (using CIFAR-10 stats by default for now, or Config logic)
    # We should use the normalization of the *Model's training data*.
    # Assuming model is trained on CIFAR-10 or CIFAR-100.
    
    # We reuse the test_transform logic from get_dataloaders, but we need to extract it or copy it.
    # For simplicity, let's redefine it based on Config.
    
    if Config.DATASET_NAME == 'cifar100':
        mean, std = (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
    else:
        mean, std = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
        
    transform = transforms.Compose([
        transforms.Resize((32, 32)), # Ensure size matches
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    
    if ood_dataset_name == 'svhn':
        dataset = datasets.SVHN(root=Config.DATA_DIR, split='test', download=True, transform=transform)
    elif ood_dataset_name == 'cifar100':
        dataset = datasets.CIFAR100(root=Config.DATA_DIR, train=False, download=True, transform=transform)
    elif ood_dataset_name == 'cifar10':
        dataset = datasets.CIFAR10(root=Config.DATA_DIR, train=False, download=True, transform=transform)
    elif ood_dataset_name == 'lsun':
         # LSUN is large, usually use LSUN-Resize or Classroom
         # Keeping it simple for now, maybe just SVHN/CIFAR
         raise NotImplementedError("LSUN ot supported yet")
    else:
        raise ValueError(f"Unknown OOD dataset: {ood_dataset_name}")
        
    loader = DataLoader(dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=2)
    return loader
