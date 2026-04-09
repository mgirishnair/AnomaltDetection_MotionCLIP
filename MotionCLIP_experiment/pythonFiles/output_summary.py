import os

import re



INPUT_DIR = "/scratch/mgirishnair/MotionCLIP_experiment/results/multimodal/classwiseGaussian/contrastiveSplit/final"

OUTPUT_FILE = "auroc_summary.txt"


def extract_split_name(filename):
    # example: anomaly_results_multimodalClasswiseCosine_arms.txt
    return filename.split("_")[-1].replace(".txt", "")

def extract_auroc(file_path):
    with open(file_path, "r") as f:
        for line in f:
            if "AUROC" in line:
                return float(line.strip().split(":")[1])
    return None

def main():
    results = []
    for fname in os.listdir(INPUT_DIR):
        if fname.endswith(".txt"):
            split = extract_split_name(fname)
            fpath = os.path.join(INPUT_DIR, fname)
            auroc = extract_auroc(fpath)
            if auroc is not None:
                results.append((split, auroc))

    # sort (optional)
    results.sort()
    OUTPUT_PATH = os.path.join(INPUT_DIR, OUTPUT_FILE)
    with open(OUTPUT_PATH, "w") as f:
        for split, auroc in results:
            f.write(f"{split} - {auroc:.4f}\n")
    print(f"Saved summary to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
