import torch
import torch.nn as nn
from src.model import CoarseToFineNet # We can reuse the architecture

# Simple training function for baseline
def train_baseline(model, loader, optimizer, device):
    model.train()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0
    for images, fine_labels, _ in loader: # Ignore coarse_labels
        images, fine_labels = images.to(device), fine_labels.to(device)
        optimizer.zero_grad()
        fine_out, _ = model(images) # Only care about fine_out
        loss = criterion(fine_out, fine_labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)