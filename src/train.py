import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.model import CoarseToFineNet

def train_one_epoch(model, loader, optimizer, device, lambda_coarse):
    model.train()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0
    
    for images, fine_labels, coarse_labels in loader:
        images, fine_labels, coarse_labels = images.to(device), fine_labels.to(device), coarse_labels.to(device)
        
        optimizer.zero_grad()
        fine_out, coarse_out = model(images)
        
        # Multi-task loss: Combined error
        loss_fine = criterion(fine_out, fine_labels)
        loss_coarse = criterion(coarse_out, coarse_labels)
        loss = loss_fine + (lambda_coarse * loss_coarse)
        
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        
    return total_loss / len(loader)