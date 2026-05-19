#!/usr/bin/env python
# coding: utf-8

import argparse
import pandas as pd


def read_ntu_labels(path):
    with open(path, "r", encoding="utf-8") as f:
        labels = [line.strip() for line in f if line.strip()]

    # 1-based mapping: line 1 -> class 1, line 91 -> class 91
    return {i + 1: label for i, label in enumerate(labels)}


def split_pipe_list(value):
    return [x.strip() for x in str(value).split("|") if x.strip()]


def split_pipe_scores(value):
    return [float(x.strip()) for x in str(value).split("|") if x.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", required=True)
    parser.add_argument("--ntu_labels_path", required=True)
    parser.add_argument("--output_csv", required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.input_csv)
    ntu_map = read_ntu_labels(args.ntu_labels_path)

    required_cols = [
        "ntu_class",
        "class_level_topk_babel_labels",
        "class_level_topk_raw_scores",
        "class_level_topk_corrected_scores",
        "babel_likeness_score",
        "overlap_group",
    ]

    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns: {missing}\n"
            f"Available columns: {list(df.columns)}"
        )

    rows = []

    for _, row in df.iterrows():
        ntu_id = int(row["ntu_class"])
        ntu_label = ntu_map.get(ntu_id, "")

        babel_labels = split_pipe_list(row["class_level_topk_babel_labels"])
        raw_scores = split_pipe_scores(row["class_level_topk_raw_scores"])
        corrected_scores = split_pipe_scores(row["class_level_topk_corrected_scores"])

        n = min(len(babel_labels), len(raw_scores), len(corrected_scores))

        if n == 0:
            print(f"WARNING: no BABEL matches found for NTU class {ntu_id}")
            continue

        if not (len(babel_labels) == len(raw_scores) == len(corrected_scores)):
            print(
                f"WARNING: length mismatch for NTU class {ntu_id}: "
                f"{len(babel_labels)} labels, "
                f"{len(raw_scores)} raw scores, "
                f"{len(corrected_scores)} corrected scores. "
                f"Using first {n} entries."
            )

        for i in range(n):
            rows.append({
                "ntu_id": ntu_id,
                "ntu_label": ntu_label,
                "babel_rank": i + 1,
                "babel_label": babel_labels[i],
                "raw_similarity": raw_scores[i],
                "corrected_similarity": corrected_scores[i],
                "overall_ntu_overlap_score": float(row["babel_likeness_score"]),
                "overlap_group": row["overlap_group"],
            })

    out = pd.DataFrame(rows)

    out = out[
        [
            "ntu_id",
            "ntu_label",
            "babel_rank",
            "babel_label",
            "raw_similarity",
            "corrected_similarity",
            "overall_ntu_overlap_score",
            "overlap_group",
        ]
    ]

    out.to_csv(args.output_csv, index=False)

    print("Saved:", args.output_csv)
    print("Rows written:", len(out))


if __name__ == "__main__":
    main()
