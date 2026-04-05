# config.py — all hyperparameters in one place

SEED        = 42
BATCH_SIZE  = 1        # keeping 1 to match original; increase for speed

# Training
EPOCHS      = 10
LR          = 1e-4
BETAS       = (0.9, 0.999)
LR_FACTOR   = 0.1
LR_PATIENCE = 3

# Distillation
TEMPERATURE = 20       # higher T → softer probability distributions

# Attack evaluation
EPSILONS = [0, 0.007, 0.01, 0.02, 0.03, 0.05, 0.1, 0.2, 0.3]
