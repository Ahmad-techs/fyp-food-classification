import torch
import torch.nn as nn
import numpy as np
import random
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

def set_seeds(seed: int = 42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class EarlyStopping:
    def __init__(self, patience: int = 5):
        self.patience = patience
        self.counter = 0
        self.best = 0.0

    def step(self, val_acc: float) -> bool:
        if val_acc > self.best:
            self.best = val_acc
            self.counter = 0
            return False
        self.counter += 1
        if self.counter >= self.patience:
            print(f'  Early stopping — no improvement for {self.patience} epochs')
            return True
        return False

@torch.no_grad()
def _eval_model(model, loader, device, model_type):
    model.eval()
    f_correct = c_correct = total = 0
    for imgs, fl, cl in loader:
        imgs, fl, cl = imgs.to(device), fl.to(device), cl.to(device)
        c_out, f_out = model(imgs)
        f_correct += (f_out.argmax(1) == fl).sum().item()
        if model_type == 'dual':
            c_correct += (c_out.argmax(1) == cl).sum().item()
        total += fl.size(0)
    
    f_acc = 100 * f_correct / total
    c_acc = 100 * c_correct / total if model_type == 'dual' else 0.0
    return f_acc, c_acc

def _run_phase(model, train_loader, val_loader, device, epochs, lr, save_path, model_type, lam, patience, history, phase_name):
    set_seeds(42)
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    es = EarlyStopping(patience=patience)
    best_acc = history.get('best_fine_acc', 0.0)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for imgs, fl, cl in train_loader:
            imgs, fl, cl = imgs.to(device), fl.to(device), cl.to(device)
            optimizer.zero_grad()
            
            c_out, f_out = model(imgs)
            if model_type == 'single':
                loss = criterion(f_out, fl)
            else:
                loss = lam * criterion(c_out, cl) + (1 - lam) * criterion(f_out, fl)

            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        scheduler.step()
        f_acc, c_acc = _eval_model(model, val_loader, device, model_type)
        
        print(f'  {phase_name} Ep{epoch+1:02d}/{epochs} loss={running_loss/len(train_loader):.4f} Fine={f_acc:.2f}% Coarse={c_acc:.2f}%')

        if f_acc > best_acc:
            best_acc = f_acc
            history['best_fine_acc'] = best_acc
            torch.save({'model_state_dict': model.state_dict(), 'best_fine_acc': best_acc}, save_path)
            print(f'  ✓ Checkpoint saved (best={best_acc:.2f}%)')

        if es.step(f_acc): break
    return history

def train_model(model, train_loader, val_loader, device, save_path, model_type='single', lam=0.0, patience=5):
    history = {'best_fine_acc': 0.0}
    model.freeze_backbone()
    history = _run_phase(model, train_loader, val_loader, device, 5, 1e-3, save_path, model_type, lam, patience, history, 'P1')
    model.unfreeze_top(n=3)
    history = _run_phase(model, train_loader, val_loader, device, 15, 1e-4, save_path, model_type, lam, patience, history, 'P2')
    model.unfreeze_all()
    history = _run_phase(model, train_loader, val_loader, device, 5, 5e-5, save_path, model_type, lam, patience, history, 'P3')
    return history