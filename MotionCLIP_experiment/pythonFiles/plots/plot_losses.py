import os
import json
import argparse
import matplotlib.pyplot as plt


def plot_losses(json_path, output_dir,filename):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    history = data.get("history", [])
    if not history:
        raise ValueError("No 'history' found in the JSON file.")

    epochs = [entry["epoch"] for entry in history]
    train_loss = [entry["train_loss"] for entry in history]
    val_loss = [entry["val_loss"] for entry in history]

    os.makedirs(output_dir, exist_ok=True)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_loss, label="Train Loss")
    plt.plot(epochs, val_loss, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    out_path = os.path.join(output_dir, filename)
    plt.savefig(out_path, dpi=300)
    plt.close()

    print(f"Saved plot to: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_path", type=str, required=True, help="Path to finetune summary JSON")
    parser.add_argument("--output_dir", type=str, required=True, help="Folder to save the plot")
    parser.add_argument("--filename", type=str, required=True, help="Name of the output PNG file")
    args = parser.parse_args()

    plot_losses(args.json_path, args.output_dir, args.filename)
