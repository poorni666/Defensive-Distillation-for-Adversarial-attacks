# attacks.py — adversarial attack implementations

import torch


def fgsm_attack(inputs: torch.Tensor, epsilon: float, data_grad: torch.Tensor) -> torch.Tensor:
    """
    Fast Gradient Sign Method (FGSM) — Goodfellow et al., 2014.

    Perturbs each pixel by epsilon in the direction that maximises the loss:
        x_adv = x + epsilon * sign( ∇_x J(θ, x, y) )

    Args:
        inputs:    original input tensor (must be in [0, 1])
        epsilon:   perturbation magnitude
        data_grad: gradient of loss w.r.t. inputs

    Returns:
        Adversarial example clamped to [0, 1].

    Note: clamping to [0, 1] is correct here because the dataloader uses
    Normalize((0,), (1,)) which is a no-op — pixel values stay in [0, 1]
    after ToTensor().  If you switch to proper MNIST normalisation
    (mean=0.1307, std=0.3081), update the clamp bounds accordingly.
    """
    perturbed = inputs + epsilon * data_grad.sign()
    return torch.clamp(perturbed, 0.0, 1.0)
