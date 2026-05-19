import os
import pickle

PKL_LIST_FILE = "/scratch/mgirishnair/Thesis/smpl_pkl_files_120.txt"

def main():
    if not os.path.isfile(PKL_LIST_FILE):
        raise FileNotFoundError(f"PKL list file not found: {PKL_LIST_FILE}")

    labels = set()
    total_files = 0
    missing_files = 0
    failed_files = 0

    with open(PKL_LIST_FILE, "r", encoding="utf-8") as f:
        for line in f:
            pkl_path = line.strip()
            if not pkl_path:
                continue

            total_files += 1

            if not os.path.isfile(pkl_path):
                print(f"[MISSING] {pkl_path}")
                missing_files += 1
                continue

            try:
                with open(pkl_path, "rb") as pf:
                    data = pickle.load(pf)

                if "label" not in data:
                    print(f"[NO LABEL KEY] {pkl_path}")
                    failed_files += 1
                    continue

                labels.add(str(data["label"]).strip())

            except Exception as e:
                print(f"[FAILED] {pkl_path} -> {e}")
                failed_files += 1

    print("\n=== UNIQUE LABELS FOUND ===")
    for label in sorted(labels):
        print(label)

    print("\n=== SUMMARY ===")
    print(f"Total listed files : {total_files}")
    print(f"Missing files      : {missing_files}")
    print(f"Failed files       : {failed_files}")
    print(f"Unique labels      : {len(labels)}")

if __name__ == "__main__":
    main()
