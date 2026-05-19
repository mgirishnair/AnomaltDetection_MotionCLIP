import os
import pickle
from tqdm import tqdm
from NTU_actions import NTU_ACTIONS  # your mapping

PKL_LIST_FILE = "/scratch/mgirishnair/Thesis/smpl_pkl_files_120.txt"
OUTPUT_FILE = "missing_labels.txt"

def main():
    with open(PKL_LIST_FILE, "r") as f:
        pkl_paths = [line.strip() for line in f if line.strip()]

    dataset_labels = set()

    for pkl_path in tqdm(pkl_paths, desc="Scanning labels"):
        try:
            with open(pkl_path, "rb") as pf:
                data = pickle.load(pf)
            dataset_labels.add(str(data["label"]).strip())
        except:
            continue

    dict_labels = set(NTU_ACTIONS.keys())

    missing = sorted(dataset_labels - dict_labels)
    extra = sorted(dict_labels - dataset_labels)

    # write to file
    with open(OUTPUT_FILE, "w") as f:
        f.write("=== LABELS IN DATA BUT NOT IN DICTIONARY ===\n")
        for m in missing:
            f.write(f"{m}\n")

        f.write("\n=== LABELS IN DICTIONARY BUT NOT IN DATA ===\n")
        for e in extra:
            f.write(f"{e}\n")

    print(f"\nSaved mismatches to {OUTPUT_FILE}")
    print(f"Missing labels: {len(missing)}")
    print(f"Extra labels: {len(extra)}")

if __name__ == "__main__":
    main()
