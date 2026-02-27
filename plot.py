# Copyright 2025-2026 Muhammad Nizwa. All rights reserved.

import numpy as np
import matplotlib.pyplot as plt


def plot_loss(history):
    plt.figure(figsize=(8, 5))
    best_epoch = np.argmin(history["val_loss"])
    plt.axvline(
        x=best_epoch,
        color="green",
        linestyle="--",
        linewidth=1,
        label=f"Best Epoch ({np.min(history["val_loss"]):.4f} Val Loss)",
    )
    plt.grid(True, linestyle="--", linewidth=0.6, alpha=0.8)
    plt.plot(history["train_loss"], label="Train Loss", color="red")
    plt.plot(history["val_loss"], label="Val Loss", color="orange")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Training Loss")
    plt.legend()
    plt.show()


def plot_perplexity(history):
    plt.figure(figsize=(8, 5))
    best_epoch = np.argmin(history["val_perplexity"])
    plt.axvline(
        x=best_epoch,
        color="green",
        linestyle="--",
        linewidth=1,
        label=f"Best Epoch ({np.min(history["val_perplexity"]):.4f} Perplexity)",
    )
    plt.grid(True, linestyle="--", linewidth=0.6, alpha=0.8)
    plt.plot(history["train_perplexity"], label="Train PPL", color="blue")
    plt.plot(history["val_perplexity"], label="Val PPL", color="green")
    plt.xlabel("Epoch")
    plt.ylabel("Perplexity")
    plt.title(f"Perplexity Metrics")
    plt.legend()
    plt.show()