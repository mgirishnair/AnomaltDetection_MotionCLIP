#!/usr/bin/env python
# coding: utf-8

import argparse
import pandas as pd


def read_ntu_labels(path):
    with open(path, "r", encoding="utf-8") as f:
        labels = [line.strip() for line in f if line.strip()]

    # Make 1-based mapping: 1 -> first line, 2 -> second line, ...
    return {i + 1: label for i, label in enumerate(labels)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", required=True)
    parser.add_argument("--ntu_labels_path", required=True)
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--ntu_id_col", default="ntu_class")
    args = parser.parse_args()

    df = pd.read_csv(args.input_csv)
    ntu_map = read_ntu_labels(args.ntu_labels_path)

    if args.ntu_id_col not in df.columns:
        raise ValueError(f"Column '{args.ntu_id_col}' not found. Available columns: {list(df.columns)}")

    df[args.ntu_id_col] = df[args.ntu_id_col].astype(int)

    df.insert(
        loc=df.columns.get_loc(args.ntu_id_col) + 1,
        column="ntu_label",
        value=df[args.ntu_id_col].map(ntu_map),
    )

    missing = df[df["ntu_label"].isna()][args.ntu_id_col].unique()
    if len(missing) > 0:
        print("WARNING: These NTU IDs were not found in ntu_labels.txt:", missing.tolist())

    df.to_csv(args.output_csv, index=False)
    print("Saved:", args.output_csv)


if __name__ == "__main__":
    main()
