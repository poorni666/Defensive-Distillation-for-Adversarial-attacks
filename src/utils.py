# utils.py — shared plotting helpers

import numpy as np
import matplotlib.pyplot as plt


def plot_loss_curves(train_loss: list, val_loss: list, title: str = ""):
    """Plot training and validation loss over epochs."""
    epochs = np.arange(1, len(train_loss) + 1)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(epochs, train_loss, "*-", label="Train loss")
    ax.plot(epochs, val_loss,   "o-", label="Val loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    if title:
        ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    plt.show()


def plot_accuracy_vs_epsilon(epsilons: list, accuracies: list, title: str = "FGSM robustness"):
    """Plot model accuracy as a function of FGSM epsilon."""
    plt.figure(figsize=(5, 4))
    plt.plot(epsilons, accuracies, "*-")
    plt.xlabel("Epsilon (ε)")
    plt.ylabel("Accuracy")
    plt.title(title)
    plt.tight_layout()
    plt.show()


def plot_defense_comparison(epsilons: list, baseline_acc: list, defense_acc: list,
                            defense_label: str = "Distilled student"):
    """
    Side-by-side accuracy curves: baseline (no defence) vs defended model.
    Shaded area between the two curves highlights the robustness gain.
    """
    fig, ax = plt.subplots(figsize=(7, 4))

    ax.plot(epsilons, baseline_acc, "*-",  color="#E24B4A", linewidth=2,
            label="Baseline (no defence)")
    ax.plot(epsilons, defense_acc,  "o--", color="#1D9E75", linewidth=2,
            label=defense_label)

    ax.fill_between(epsilons, baseline_acc, defense_acc,
                    where=[d >= b for b, d in zip(baseline_acc, defense_acc)],
                    alpha=0.15, color="#1D9E75", label="Robustness gain")
    ax.fill_between(epsilons, baseline_acc, defense_acc,
                    where=[d < b for b, d in zip(baseline_acc, defense_acc)],
                    alpha=0.15, color="#E24B4A", label="Accuracy cost")

    ax.set_xlabel("Epsilon (ε)", fontsize=12)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title("FGSM robustness: baseline vs defensive distillation", fontsize=13)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.show()


def plot_adversarial_examples(epsilons: list, examples: list):
    """
    Grid: rows = epsilon values, columns = adversarial examples.
    Each cell title shows 'original_label → predicted_label'.
    """
    n_rows = len(epsilons)
    n_cols = max((len(ex) for ex in examples), default=1)
    if n_cols == 0:
        print("No adversarial examples to display.")
        return

    cnt = 0
    plt.figure(figsize=(2 * n_cols, 2 * n_rows))
    for i, row_examples in enumerate(examples):
        for j, (orig, adv, img) in enumerate(row_examples):
            cnt += 1
            plt.subplot(n_rows, n_cols, cnt)
            plt.xticks([], [])
            plt.yticks([], [])
            if j == 0:
                plt.ylabel(f"ε={epsilons[i]:.3f}", fontsize=11)
            plt.title(f"{orig} → {adv}", fontsize=10)
            plt.imshow(img, cmap="gray")
    plt.tight_layout()
    plt.show()
