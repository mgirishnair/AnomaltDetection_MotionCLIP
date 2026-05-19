from pathlib import Path
import re
import sys

def extract_best_trial(text: str):
    split_match = re.search(r'^\s*split\s*:\s*(\S+)\s*$', text, re.MULTILINE)
    split_name = split_match.group(1) if split_match else "global_all_classes"

    #if not split_match:
        #return None

    #split_name = split_match.group(1)

    block_match = re.search(
        r'Best trial:\s*'
        r'(?:.*?\n)*?'
        r'\s*params:\s*\n'
        r'((?:\s+[A-Za-z0-9_]+\s*:\s*.+\n)+)',
        text,
        re.MULTILINE
    )
    if not block_match:
        return None

    params_block = block_match.group(1)

    params = {}
    for line in params_block.strip().splitlines():
        m = re.match(r'^\s*([A-Za-z0-9_]+)\s*:\s*(.+?)\s*$', line)
        if m:
            key, value = m.group(1), m.group(2)
            params[key] = value

    needed = ["lr_encoder", "contrastive_temp", "n_samples_per_class", "weight_decay"]
    if not all(k in params for k in needed):
        return None

    return {
        "split": split_name,
        "lr_encoder": params["lr_encoder"],
        "contrastive_temp": params["contrastive_temp"],
        "n_samples_per_class": params["n_samples_per_class"],
        "weight_decay": params["weight_decay"],
    }


def main():
    if len(sys.argv) != 3:
        print("Usage: python extract_optuna_best.py <out_dir> <output_txt>")
        sys.exit(1)

    out_dir = Path(sys.argv[1])
    output_txt = Path(sys.argv[2])

    if not out_dir.is_dir():
        print(f"Error: directory not found: {out_dir}")
        sys.exit(1)

    results = []
    failed_files = []

    for out_file in sorted(out_dir.glob("*.out")):
        text = out_file.read_text(encoding="utf-8", errors="ignore")
        result = extract_best_trial(text)

        if result is None:
            failed_files.append(out_file.name)
            continue

        results.append(result)

    if not results:
        print("Error: no valid best-trial blocks found in any .out files.")
        sys.exit(1)

    results.sort(key=lambda x: x["split"])

    with output_txt.open("w", encoding="utf-8") as f:
        for r in results:
            f.write(
                f'{r["split"]}: '
                f'lr_encoder={r["lr_encoder"]}, '
                f'contrastive_temp={r["contrastive_temp"]}, '
                f'n_samples_per_class={r["n_samples_per_class"]}, '
                f'weight_decay={r["weight_decay"]}\n'
            )

    print(f"Wrote {len(results)} splits to: {output_txt}")

    if failed_files:
        print("\nWarning: could not parse these files:")
        for name in failed_files:
            print(f"  {name}")


if __name__ == "__main__":
    main()
