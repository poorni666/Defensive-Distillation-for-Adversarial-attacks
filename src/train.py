# train.py — training loop and FGSM evaluation

import torch
import torch.nn as nn
import torch.nn.functional as F

from attacks import fgsm_attack


def fit(model, device, optimizer, scheduler, train_loader, val_loader,
        temperature: float = 1.0, epochs: int = 10):
    """
    Train a model with hard (integer) labels and temperature-scaled softmax.
    Args:
        temperature: softmax temperature (T=1 → standard; T>1 → softer).
                     Used for the teacher in defensive distillation.

    Returns:
        (train_losses, val_losses) — one value per epoch.
    """
    criterion = nn.NLLLoss()
    train_losses, val_losses = [], []

    for epoch in range(epochs):
        # ── Training phase ────────────────────────────────────────────────
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            log_probs = F.log_softmax(model(inputs) / temperature, dim=1)
            loss = criterion(log_probs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        # ── Validation phase ──────────────────────────────────────────────
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                log_probs = F.log_softmax(model(inputs) / temperature, dim=1)
                val_loss += criterion(log_probs, labels).item()

        avg_train = running_loss / len(train_loader)
        avg_val   = val_loss    / len(val_loader)
        scheduler.step(avg_val)

        print(f"Epoch {epoch+1:>2}/{epochs}  "
              f"Loss: {avg_train:.4f}  Val Loss: {avg_val:.4f}")
        train_losses.append(avg_train)
        val_losses.append(avg_val)

    return train_losses, val_losses


def test(model, device, test_loader, epsilon: float, temperature: float = 1.0):
    """
    Evaluate model robustness against FGSM at a given epsilon.

    Only samples that are correctly classified *before* the attack are
    included in the denominator (matching the standard FGSM evaluation
    protocol).

    Args:
        epsilon:     FGSM perturbation magnitude.
        temperature: must match the temperature used during training.

    Returns:
        (accuracy, adversarial_examples)
        adversarial_examples: list of (original_label, predicted_label, image_array)
    """
    model.eval()
    correct      = 0
    adv_examples = []

    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        data.requires_grad = True

        log_probs = F.log_softmax(model(data) / temperature, dim=1)
        init_pred = log_probs.max(1, keepdim=True)[1]

        # Skip samples already misclassified
        if init_pred.item() != target.item():
            continue

        loss = F.nll_loss(log_probs, target)
        model.zero_grad()
        loss.backward()

        perturbed = fgsm_attack(data, epsilon, data.grad.data)

        with torch.no_grad():
            log_probs_adv = F.log_softmax(model(perturbed) / temperature, dim=1)
        final_pred = log_probs_adv.max(1, keepdim=True)[1]

        if final_pred.item() == target.item():
            correct += 1
            if epsilon == 0 and len(adv_examples) < 5:
                adv_examples.append((
                    init_pred.item(), final_pred.item(),
                    perturbed.squeeze().detach().cpu().numpy()
                ))
        else:
            if len(adv_examples) < 5:
                adv_examples.append((
                    init_pred.item(), final_pred.item(),
                    perturbed.squeeze().detach().cpu().numpy()
                ))

    accuracy = correct / float(len(test_loader))
    print(f"ε={epsilon:.3f}  Accuracy: {correct}/{len(test_loader)} = {accuracy:.4f}")
    return accuracy, adv_examples
