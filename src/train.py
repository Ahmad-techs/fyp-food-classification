# src/train.py
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
    torch.backends.cudnn.benchmark     = False
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class EarlyStopping:
    def __init__(self, patience: int = 5):
        self.patience = patience
        self.counter  = 0
        self.best     = 0.0

    def step(self, val_acc: float) -> bool:
        if val_acc > self.best:
            self.best = val_acc; self.counter = 0; return False
        self.counter += 1
        if self.counter >= self.patience:
            print(f'  Early stopping — no improvement for {self.patience} epochs')
            return True
        return False

@torch.no_grad()
def _eval_single(model, loader, device):
    model.eval(); correct = total = 0
    for imgs, fl, _ in loader:
        imgs, fl = imgs.to(device), fl.to(device)
        out      = model(imgs)
        correct += (out.argmax(1) == fl).sum().item()
        total   += fl.size(0)
    return 100 * correct / total

@torch.no_grad()
def _eval_dual(model, loader, device):
    model.eval(); f_correct = c_correct = total = 0
    for imgs, fl, cl in loader:
        imgs, fl, cl = imgs.to(device), fl.to(device), cl.to(device)
        c_out, f_out = model(imgs)
        f_correct   += (f_out.argmax(1) == fl).sum().item()
        c_correct   += (c_out.argmax(1) == cl).sum().item()
        total       += fl.size(0)
    return 100 * f_correct / total, 100 * c_correct / total

def _run_phase(model, train_loader, val_loader, device,
               epochs, lr, save_path, model_type, lam,
               patience, history, phase_name):
    set_seeds(42)
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW([p for p in model.parameters() if p.requires_grad],
                       lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    es        = EarlyStopping(patience=patience)
    best_acc  = history.get('best_fine_acc', 0.0)

    for epoch in range(epochs):
        model.train(); running_loss = 0.0
        for batch in train_loader:
            imgs = batch[0].to(device)
            fl   = batch[1].to(device)
            optimizer.zero_grad()
            if model_type == 'single':
                loss = criterion(model(imgs), fl)
            else:
                cl           = batch[2].to(device)
                c_out, f_out = model(imgs)
                loss = lam * criterion(c_out, cl) + (1 - lam) * criterion(f_out, fl)
            running_loss += loss.item()
            loss.backward()
            optimizer.step()
        scheduler.step()

        avg_loss = running_loss / len(train_loader)
        if model_type == 'single':
            top1 = _eval_single(model, val_loader, device)
            print(f'  {phase_name} Ep{epoch+1:02d}/{epochs} loss={avg_loss:.4f} Top-1={top1:.2f}%')
        else:
            top1, ctop1 = _eval_dual(model, val_loader, device)
            print(f'  {phase_name} Ep{epoch+1:02d}/{epochs} loss={avg_loss:.4f} Fine={top1:.2f}% Coarse={ctop1:.2f}%')

        history.setdefault('val_fine_top1', []).append(top1)

        if top1 > best_acc:
            best_acc = top1
            history['best_fine_acc'] = best_acc
            torch.save({'model_state_dict': model.state_dict(),
                        'best_fine_acc': best_acc,
                        'history': history,
                        'lambda_coarse': lam}, save_path)
            print(f'  ✓ Checkpoint saved (best={best_acc:.2f}%)')
        if es.step(top1): break
    return history

def train_model(model, train_loader, val_loader, device,
                save_path, model_type='single', lam=0.0, patience=5):
    """3-phase training — identical for all 4 experimental conditions."""
    set_seeds(42)
    history = {}
    print('=== Phase 1: Heads only | 5 epochs | lr=1e-3 ===')
    model.freeze_backbone()
    history = _run_phase(model, train_loader, val_loader, device,
                         5, 1e-3, save_path, model_type, lam, patience, history, 'P1')
    print('=== Phase 2: Top-3 blocks unfrozen | 15 epochs | lr=1e-4 ===')
    model.unfreeze_top(n=3)
    history = _run_phase(model, train_loader, val_loader, device,
                         15, 1e-4, save_path, model_type, lam, patience, history, 'P2')
    print('=== Phase 3: Full backbone | 5 epochs | lr=5e-5 ===')
    model.unfreeze_all()
    history = _run_phase(model, train_loader, val_loader, device,
                         5, 5e-5, save_path, model_type, lam, patience, history, 'P3')
    print(f'Training complete. Best val Top-1: {history["best_fine_acc"]:.2f}%')
    return history