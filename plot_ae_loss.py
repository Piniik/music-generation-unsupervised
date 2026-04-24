import json
from pathlib import Path

import matplotlib.pyplot as plt

history_path = Path("checkpoints/ae_smoke_history.json")
out_path = Path("outputs/plots/ae_reconstruction_loss.png")

with open(history_path, "r", encoding="utf-8") as f:
    history = json.load(f)

epochs = history["epoch"]
train_loss = history["train_loss"]
val_loss = history["val_loss"]

out_path.parent.mkdir(parents=True, exist_ok=True)

plt.figure(figsize=(8, 5))
plt.plot(epochs, train_loss, marker="o", label="Train Reconstruction Loss")
plt.plot(epochs, val_loss, marker="o", label="Validation Reconstruction Loss")
plt.xlabel("Epoch")
plt.ylabel("Reconstruction Loss")
plt.title("Task 1 LSTM Autoencoder Reconstruction Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(out_path, dpi=200)

print(f"Saved plot to: {out_path}")