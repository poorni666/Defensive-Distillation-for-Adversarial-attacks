# defense.py — defensive distillation (Papernot et al., 2016)
#
# The idea: train a teacher model F at high temperature T, use its soft
# probability outputs as training targets for a smaller student F'.
# The smooth targets reduce the gradient information available to an attacker,
# making FGSM perturbations less effective.
#
# Pipeline:
#   1. Train teacher  Net    on MNIST hard labels at temperature T.
#   2. Generate soft labels  p_T = softmax(F(x) / T)  for every training sample.
#   3. Train student  NetF1  on those soft labels at temperature T.
#   4. At test time evaluate the student at T=1 (sharp predictions).

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt

from models import Net, NetF1
from train import test


# ── Soft-label dataset ────────────────────────────────────────────────────────

class SoftLabelDataset(Dataset):
    """
    Pre-computes soft labels from a teacher model and stores them in memory.
    Args:
        loader:      DataLoader over the training set (any batch size).
        teacher:     trained teacher model (must already be on `device`).
        device:      compute device.
        temperature: T used to smooth the teacher's output distribution.
    """
    def __init__(self, loader: DataLoader, teacher: nn.Module,
                 device: torch.device, temperature: float):
        teacher.eval()
        self.samples = []
        with torch.no_grad():
            for inputs, _ in loader:
                inputs = inputs.to(device)
                soft = F.softmax(teacher(inputs) / temperature, dim=1)
                for i in range(inputs.size(0)):
                    self.samples.append((inputs[i].cpu(), soft[i].cpu()))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


# ── Student training helpers ──────────────────────────────────────────────────

def _fit_student(model, device, optimizer, scheduler,
                 soft_train_loader, val_loader,
                 temperature: float, epochs: int):
    """
    Train the student network.

    Training loss : KL-divergence between student log-probs and teacher
                    soft-prob targets.  KLDivLoss is the correct choice here —
                    NLLLoss expects integer class indices, not probability
                    vectors (another bug in the original code).

    Validation loss: standard NLLLoss against hard labels (from val_loader)
                     so the reported metric is interpretable.
    """
    nll = nn.NLLLoss()
    train_losses, val_losses = [], []

    for epoch in range(epochs):
        # Training on soft labels
        model.train()
        running = 0.0
        for inputs, soft_labels in soft_train_loader:
            inputs, soft_labels = inputs.to(device), soft_labels.to(device)
            optimizer.zero_grad()
            log_probs = F.log_softmax(model(inputs) / temperature, dim=1)
            # KL( soft_labels || student ) — correct loss for soft targets
            loss = F.kl_div(log_probs, soft_labels, reduction='batchmean')
            loss.backward()
            optimizer.step()
            running += loss.item()

        # Validation on hard labels
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                log_probs = F.log_softmax(model(inputs) / temperature, dim=1)
                val_loss += nll(log_probs, labels).item()

        avg_train = running  / len(soft_train_loader)
        avg_val   = val_loss / len(val_loader)
        scheduler.step(avg_val)
        print(f"Epoch {epoch+1:>2}/{epochs}  "
              f"KL Loss: {avg_train:.5f}  Val NLL: {avg_val:.4f}")
        train_losses.append(avg_train)
        val_losses.append(avg_val)

    return train_losses, val_losses


# ── Main defense function ─────────────────────────────────────────────────────

def defense(device, train_loader, val_loader, test_loader,
            epochs: int, temperature: float, epsilons: list):
    """
    Full defensive-distillation pipeline.

    Steps:
        1. Train teacher  (Net)   with temperature T on hard labels.
        2. Build SoftLabelDataset using the trained teacher.
        3. Train student  (NetF1) with temperature T on soft labels.
        4. Evaluate the student at T=1 under FGSM across all epsilons.

    Args:
        temperature: distillation temperature T.  T=1 → no smoothing.
                     Try T=20 for a meaningful defence effect.
    """
    from config import LR, BETAS, LR_FACTOR, LR_PATIENCE

    # ── 1. Teacher ────────────────────────────────────────────────────────
    print("=" * 60)
    print(f"Training teacher (Net) at T={temperature}")
    print("=" * 60)
    teacher = Net().to(device)
    opt_t   = optim.Adam(teacher.parameters(), lr=LR, betas=BETAS)
    sch_t   = optim.lr_scheduler.ReduceLROnPlateau(
                  opt_t, mode='min', factor=LR_FACTOR, patience=LR_PATIENCE)

    from train import fit
    loss_t, val_loss_t = fit(teacher, device, opt_t, sch_t,
                              train_loader, val_loader,
                              temperature=temperature, epochs=epochs)

    _plot_loss(loss_t, val_loss_t, epochs, title="Teacher (Net)")

    # ── 2. Soft labels ────────────────────────────────────────────────────
    print("\nGenerating soft labels from teacher…")
    soft_dataset = SoftLabelDataset(train_loader, teacher, device, temperature)
    soft_loader  = DataLoader(soft_dataset, batch_size=1, shuffle=True)
    print(f"Soft-label dataset size: {len(soft_dataset)}")

    # ── 3. Student ────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"Training student (NetF1) on soft labels at T={temperature}")
    print("=" * 60)
    student = NetF1().to(device)
    opt_s   = optim.Adam(student.parameters(), lr=LR, betas=BETAS)
    sch_s   = optim.lr_scheduler.ReduceLROnPlateau(
                  opt_s, mode='min', factor=LR_FACTOR, patience=LR_PATIENCE)

    loss_s, val_loss_s = _fit_student(student, device, opt_s, sch_s,
                                       soft_loader, val_loader,
                                       temperature=temperature, epochs=epochs)

    _plot_loss(loss_s, val_loss_s, epochs, title="Student (NetF1)")

    # ── 4. Evaluate at T=1 ────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Evaluating distilled student under FGSM (T=1 at inference)")
    print("=" * 60)
    accuracies, examples = [], []
    for eps in epsilons:
        acc, ex = test(student, device, test_loader, eps, temperature=1.0)
        accuracies.append(acc)
        examples.append(ex)

    _plot_accuracy(epsilons, accuracies, title="Distilled student — FGSM robustness")
    _plot_examples(epsilons, examples)

    return student, accuracies, examples


# ── Plotting helpers ──────────────────────────────────────────────────────────

def _plot_loss(train_loss, val_loss, epochs, title=""):
    fig, ax = plt.subplots(figsize=(5, 4))
    x = np.arange(1, epochs + 1)
    ax.plot(x, train_loss, "*-", label="Train loss")
    ax.plot(x, val_loss,   "o-", label="Val loss")
    ax.set_xlabel("Epoch")
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    plt.show()


def _plot_accuracy(epsilons, accuracies, title=""):
    plt.figure(figsize=(5, 4))
    plt.plot(epsilons, accuracies, "*-")
    plt.xlabel("Epsilon")
    plt.ylabel("Accuracy")
    plt.title(title)
    plt.tight_layout()
    plt.show()


def _plot_examples(epsilons, examples):
    n_eps = len(epsilons)
    n_ex  = max(len(ex) for ex in examples) if examples else 1
    if n_ex == 0:
        return
    cnt = 0
    plt.figure(figsize=(8, 10))
    for i, eps_examples in enumerate(examples):
        for j, (orig, adv, img) in enumerate(eps_examples):
            cnt += 1
            plt.subplot(n_eps, n_ex, cnt)
            plt.xticks([], [])
            plt.yticks([], [])
            if j == 0:
                plt.ylabel(f"ε={epsilons[i]}", fontsize=12)
            plt.title(f"{orig}→{adv}", fontsize=10)
            plt.imshow(img, cmap="gray")
    plt.tight_layout()
    plt.show()
