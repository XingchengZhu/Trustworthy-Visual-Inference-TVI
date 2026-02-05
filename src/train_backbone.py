import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os
from src.config import Config
from src.dataset import get_dataloaders
from src.model import ResNetBackbone

def train_one_epoch(model, loader, criterion, center_loss_func, optimizer, optimizer_center, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    # Weight for Center Loss
    weight_center = Config.CENTER_LOSS_WEIGHT
    
    for images, labels in tqdm(loader, desc="Training", leave=False):
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        optimizer_center.zero_grad()
        
        # Model returns: spatial_features, flat_features, logits
        _, features, logits = model(images)
        
        loss_cls = criterion(logits, labels)
        loss_center = center_loss_func(features, labels)
        
        loss = loss_cls + weight_center * loss_center
        
        loss.backward()
        optimizer.step()
        # Create a new optimizer for center loss or just step it manually if it were simple param.
        # But usually we use an optimizer.
        for param in center_loss_func.parameters():
            # Gradients for centers are computed in backward()
            # We need to update them. standard SGD for centers.
            # Usually lr for centers is 0.5
            pass
        optimizer_center.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(logits.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
    return running_loss / len(loader), 100 * correct / total

def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Validation", leave=False):
            images, labels = images.to(device), labels.to(device)
            # Unpack 3 values
            _, _, logits = model(images)
            loss = criterion(logits, labels)
            
            running_loss += loss.item()
            _, predicted = torch.max(logits.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    return running_loss / len(loader), 100 * correct / total

import matplotlib.pyplot as plt

def plot_metrics(train_losses, train_accs, val_losses, val_accs):
    epochs = range(1, len(train_losses) + 1)
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'b-', label='Train Loss')
    plt.plot(epochs, val_losses, 'r-', label='Val Loss')
    plt.title('Loss Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accs, 'b-', label='Train Acc')
    plt.plot(epochs, val_accs, 'r-', label='Val Acc')
    plt.title('Accuracy Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    
    plt.tight_layout()
    # Save to dataset specific directory
    save_dir = os.path.join(Config.RESULTS_DIR, Config.DATASET_NAME)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    plt.savefig(os.path.join(save_dir, 'training_metrics.png'))
    plt.close()

import argparse

def main():
    parser = argparse.ArgumentParser(description="Train TVI Backbone")
    parser.add_argument("--config", type=str, default="conf/cifar10.json", help="Path to config file")
    parser.add_argument("--log_suffix", type=str, default=None, help="Suffix for log file")
    args = parser.parse_args()
    
    # Load Config
    Config.load_config(args.config)
    
    # Set Seeds
    import numpy as np
    torch.manual_seed(Config.SEED)
    np.random.seed(Config.SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(Config.SEED)
    
    if not os.path.exists(Config.Checkpoints_DIR):
        os.makedirs(Config.Checkpoints_DIR)
        
    # Ensure base results dir exists
    if not os.path.exists(Config.RESULTS_DIR):
        os.makedirs(Config.RESULTS_DIR)
        
    # Dataset specific results will be handled in plot_metrics, but good to ensure here too if we save other things
    results_dir = os.path.join(Config.RESULTS_DIR, Config.DATASET_NAME)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
        
    from src.utils import setup_logger
    log_name = f"experiment_{args.log_suffix}" if args.log_suffix else "experiment"
    logger = setup_logger(results_dir, name=log_name)
    
    device = torch.device(Config.DEVICE)
    logger.info(f"Using device: {device}")
    logger.info(f"Starting Training on {Config.DATASET_NAME}")
    
    train_loader, _, test_loader = get_dataloaders()
    
    # Use Config.NUM_CLASSES
    from src.loss import CenterLoss
    
    # Model Setup
    from src.model import ResNetBackbone
    model = ResNetBackbone(num_classes=Config.NUM_CLASSES).to(device)
    
    criterion = nn.CrossEntropyLoss()
    center_loss = CenterLoss(num_classes=Config.NUM_CLASSES, feat_dim=model.num_features, use_gpu=True)
    center_loss = center_loss.to(device)
    
    optimizer = optim.SGD(model.parameters(), lr=Config.LR, momentum=Config.MOMENTUM, weight_decay=Config.WEIGHT_DECAY)
    optimizer_center = optim.SGD(center_loss.parameters(), lr=0.5) # Alpha usually 0.5
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Config.EPOCHS)
    
    best_acc = 0.0
    
    # Lists for plotting
    train_losses, train_accs = [], []
    val_losses, val_accs = [], []
    
    for epoch in range(Config.EPOCHS):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, center_loss, optimizer, optimizer_center, device)
        val_loss, val_acc = validate(model, test_loader, criterion, device)
        
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        scheduler.step()
        
        # Log to file
        logger.info(f"Epoch {epoch+1}/{Config.EPOCHS} | Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}%")
        
        if val_acc > best_acc:
            best_acc = val_acc
            ckpt_name = f"best_resnet18_{Config.DATASET_NAME}.pth"
            torch.save(model.state_dict(), os.path.join(Config.Checkpoints_DIR, ckpt_name))
            logger.info(f"Model Saved: {ckpt_name} (New Best Acc/Val: {best_acc:.2f}%)")
            
        # Plot every epoch or so
        plot_metrics(train_losses, train_accs, val_losses, val_accs)
            
    logger.info(f"Training Complete. Best Accuracy: {best_acc:.2f}%")

if __name__ == "__main__":
    main()
