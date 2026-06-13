import torch
import torch.nn as nn
from src.model import CoarseToFineNet # We can reuse the architecture

# Simple training function for baseline
def train_baseline(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    
    # Add a print to know it started
    print("Starting training epoch...")
    
    for i, (images, labels) in enumerate(loader):
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels) # Ensure criterion is defined
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        # Print progress every 20 batches
        if (i + 1) % 20 == 0:
            print(f"Batch {i+1}/{len(loader)} - Loss: {loss.item():.4f}")
            
    return total_loss / len(loader)