# Copyright 2025-2026 Muhammad Nizwa. All rights reserved.

import matplotlib.pyplot as plt

def line_plot(x1, x2, x1_label="X1", x2_label="X2"):
    """
    Plots a simple figure with two horizontal lines at x1 and x2 values.

    Args:
    - x1 (float): first value to plot
    - x2 (float): second value to plot
    - x1_label (str): label for x1
    - x2_label (str): label for x2
    """
    plt.figure(figsize=(6, 4))
    
    # draw lines
    plt.axhline(y=x1, color="green", linestyle="--", linewidth=1, label=f"{x1_label}: {x1:.4f}")
    plt.axhline(y=x2, color="blue", linestyle="--", linewidth=1, label=f"{x2_label}: {x2:.4f}")
    
    # grid and labels
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.xlabel("X-axis")
    plt.ylabel("Value")
    plt.title("Comparison of Values")
    plt.legend()
    plt.show()