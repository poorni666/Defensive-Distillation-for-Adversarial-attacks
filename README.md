# Defensive-Distillation-for-Adversarial-attacks

The goal of this project is to study adversarial attacks on deep neural networks and evaluate their impact on classification accuracy. To defend against such attacks, this project implements **defensive distillation** as a method to improve the robustness of a neural network under adversarial perturbations.

The attack used in this project is the **Fast Gradient Sign Method (FGSM)**, a gradient-based attack that adds small perturbations to input images in order to fool the model and reduce its robustness. This project reimplements the paper *Distillation as a Defense to Adversarial Perturbations* and follows its experimental setup as a foundation for evaluating the effect of defensive distillation under FGSM attacks.

Paper reimplemented: https://arxiv.org/pdf/1511.04508

## Project Structure

```text
Defensive-Distillation-for-Adversarial-attacks/
├── data/
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── models.py
│   ├── attacks.py
│   ├── train.py
│   ├── defense.py
│   └── utils.py
├── main.ipynb
├── README.md
├── requirements.txt
