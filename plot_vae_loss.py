import json
from pathlib import Path

import matplotlib.pyplot as plt

history_path = Path("checkpoints/vae_all_chunks_history.json")
out_path = Path("outputs/plots/vae_training_curve.png")

with open(history_path, "r", encoding="utf-8") as f:
    history = json.load(f)

epochs = history["epoch"]
train_loss = history["train_loss"]
val_loss = history["val_loss"]
train_recon = history["train_recon"]
val_recon = history["val_recon"]
train_kl = history["train_kl"]
val_kl = history["val_kl"]
beta = history["beta"]

out_path.parent.mkdir(parents=True, exist_ok=True)

plt.figure(figsize=(9, 5))
plt.plot(epochs, train_loss, marker="o", label="Train Total Loss")
plt.plot(epochs, val_loss, marker="o", label="Validation Total Loss")
plt.plot(epochs, train_recon, marker="o", linestyle="--", label="Train Reconstruction")
plt.plot(epochs, val_recon, marker="o", linestyle="--", label="Validation Reconstruction")
plt.plot(epochs, train_kl, marker="o", linestyle=":", label="Train KL")
plt.plot(epochs, val_kl, marker="o", linestyle=":", label="Validation KL")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Task 2 VAE Training Curve")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(out_path, dpi=200)

print(f"Saved plot to: {out_path}")

beta_out = Path("outputs/plots/vae_beta_schedule.png")

plt.figure(figsize=(7, 4))
plt.plot(epochs, beta, marker="o", label="Beta")
plt.xlabel("Epoch")
plt.ylabel("Beta")
plt.title("VAE KL Annealing Schedule")
plt.grid(True)
plt.tight_layout()
plt.savefig(beta_out, dpi=200)

print(f"Saved beta plot to: {beta_out}")