import torch
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix

def evaluate(model, loader, device):
    model.eval()
    all_fine_true, all_fine_pred = [], []
    all_coarse_true, all_coarse_pred = [], []
    
    with torch.no_grad():
        for images, fine_labels, coarse_labels in loader:
            images = images.to(device)
            fine_out, coarse_out = model(images)
            
            # Get predictions
            fine_pred = torch.argmax(fine_out, dim=1).cpu().numpy()
            coarse_pred = torch.argmax(coarse_out, dim=1).cpu().numpy()
            
            all_fine_true.extend(fine_labels.numpy())
            all_fine_pred.extend(fine_pred)
            all_coarse_true.extend(coarse_labels.numpy())
            all_coarse_pred.extend(coarse_pred)
            
    return {
        "fine_acc": accuracy_score(all_fine_true, all_fine_pred),
        "coarse_acc": accuracy_score(all_coarse_true, all_coarse_pred),
        "fine_cm": confusion_matrix(all_fine_true, all_fine_pred)
    }