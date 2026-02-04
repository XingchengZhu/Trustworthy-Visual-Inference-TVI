import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os
from src.config import Config
from src.dataset import get_dataloaders
from src.model import ResNet18Backbone

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in tqdm(loader, desc="Training", leave=False):
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        _, logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        
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
            _, logits = model(images)
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
    plt.savefig(os.path.join(Config.RESULTS_DIR, 'training_metrics.png'))
    plt.close()

import argparse

def main():
    parser = argparse.ArgumentParser(description="Train TVI Backbone")
    parser.add_argument("--config", type=str, default="conf/cifar10.json", help="Path to config file")
    args = parser.parse_args()
    
    # Load Config
    Config.load_config(args.config)
    
    if not os.path.exists(Config.Checkpoints_DIR):
        os.makedirs(Config.Checkpoints_DIR)
        
    # Ensure base results dir exists
    if not os.path.exists(Config.RESULTS_DIR):
        os.makedirs(Config.RESULTS_DIR)
        
    # Dataset specific results will be handled in plot_metrics, but good to ensure here too if we save other things
    results_dir = os.path.join(Config.RESULTS_DIR, Config.DATASET_NAME)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
        
    device = torch.device(Config.DEVICE)
    print(f"Using device: {device}")
    
    train_loader, _, test_loader = get_dataloaders()
    
    # Use Config.NUM_CLASSES
    model = ResNet18Backbone(num_classes=Config.NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=Config.LR, momentum=Config.MOMENTUM, weight_decay=Config.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Config.EPOCHS)
    
    best_acc = 0.0
    
    # Lists for plotting
    train_losses, train_accs = [], []
    val_losses, val_accs = [], []
    
    for epoch in range(Config.EPOCHS):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, test_loader, criterion, device)
        
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        scheduler.step()
        
        print(f"Epoch {epoch+1}/{Config.EPOCHS} | Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}%")
        
        if val_acc > best_acc:
            best_acc = val_acc
            ckpt_name = f"best_resnet18_{Config.DATASET_NAME}.pth"
            torch.save(model.state_dict(), os.path.join(Config.Checkpoints_DIR, ckpt_name))
            print(f"Model Saved: {ckpt_name}")
            
        # Plot every epoch or so
        plot_metrics(train_losses, train_accs, val_losses, val_accs)
            
    print(f"Training Complete. Best Accuracy: {best_acc:.2f}%")

if __name__ == "__main__":
    main()
