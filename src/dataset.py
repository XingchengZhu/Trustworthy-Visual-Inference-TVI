import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from .config import Config

class TwoCropTransform:
    """Create two crops of the same image"""
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return [self.transform(x), self.transform(x)]

def get_dataloaders(use_contrastive=False):
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
    # Define transforms and load datasets based on Config.DATASET_NAME
    if Config.DATASET_NAME == 'cifar10':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        
        if use_contrastive:
            transform_train = TwoCropTransform(transform_train)
        
        train_dataset = datasets.CIFAR10(root=Config.DATA_DIR, train=True, download=True, transform=transform_train)
        
        # Split train into train/support if needed, but typically we use full train for support extraction
        # For simplicity, we just return the full train loader
        
        test_dataset = datasets.CIFAR10(root=Config.DATA_DIR, train=False, download=True, transform=transform_test)
        
        # Support Set: usually same as train, or a subset. 
        # We'll use the train dataset for support loader, but maybe with test transform (no aug)?
        # Ideally support set should be clean features.
        support_dataset = datasets.CIFAR10(root=Config.DATA_DIR, train=True, download=True, transform=transform_test)
    
    elif Config.DATASET_NAME == 'cifar100':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])
        
        if use_contrastive:
            transform_train = TwoCropTransform(transform_train)
            
        train_dataset = datasets.CIFAR100(root=Config.DATA_DIR, train=True, download=True, transform=transform_train)
        test_dataset = datasets.CIFAR100(root=Config.DATA_DIR, train=False, download=True, transform=transform_test)
        support_dataset = datasets.CIFAR100(root=Config.DATA_DIR, train=True, download=True, transform=transform_test)
        
    elif Config.DATASET_NAME == 'imagenet100':
        # ImageNet Standard Normalization
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        # Train: RandomResizedCrop + Flip
        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])

        # Test/Support: Resize 256 -> CenterCrop 224
        transform_test = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])
        
        if use_contrastive:
            transform_train = TwoCropTransform(transform_train)

        # Paths
        train_dir = os.path.join(Config.DATA_DIR, 'imagenet100', 'train')
        val_dir = os.path.join(Config.DATA_DIR, 'imagenet100', 'val')
        
        if not os.path.exists(train_dir):
            raise FileNotFoundError(f"ImageNet-100 train dir not found at {train_dir}")
            
        train_dataset = datasets.ImageFolder(train_dir, transform=transform_train)
        support_dataset = datasets.ImageFolder(train_dir, transform=transform_test)
        test_dataset = datasets.ImageFolder(val_dir, transform=transform_test)
        
    elif Config.DATASET_NAME == 'svhn':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970)),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970)),
        ])
        
        if use_contrastive:
            transform_train = TwoCropTransform(transform_train)
        
        train_dataset = datasets.SVHN(root=Config.DATA_DIR, split='train', download=True, transform=transform_train)
        test_dataset = datasets.SVHN(root=Config.DATA_DIR, split='test', download=True, transform=transform_test)
        support_dataset = datasets.SVHN(root=Config.DATA_DIR, split='train', download=True, transform=transform_test)

    else:
        raise ValueError(f"Unknown Dataset: {Config.DATASET_NAME}")

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
