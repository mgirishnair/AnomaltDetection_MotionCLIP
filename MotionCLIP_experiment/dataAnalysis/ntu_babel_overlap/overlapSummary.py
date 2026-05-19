#!/usr/bin/env python
# coding: utf-8

import argparse
import pandas as pd


def aggregate_long_csv(path, score_name):
    df = pd.read_csv(path)

    grouped = (
        df.sort_values(["ntu_id", "babel_rank"])
        .groupby(["ntu_id", "ntu_label"], as_index=False)
        .agg(
            overlap_score=("overall_ntu_overlap_score", "first"),
            overlap_group=("overlap_group", "first"),
        )
    )

    return grouped.rename(
        columns={
            "overlap_score": f"{score_name}_score",
            "overlap_group": f"{score_name}_group",
        }
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--motion_long_csv", required=True)
    parser.add_argument("--label_long_csv", required=True)
    parser.add_argument("--output_csv", required=True)
    args = parser.parse_args()

    motion = aggregate_long_csv(args.motion_long_csv, "motion")
    label = aggregate_long_csv(args.label_long_csv, "label")

    df = motion.merge(
        label,
        on=["ntu_id", "ntu_label"],
        how="outer",
        validate="one_to_one",
    )

    df["motion_score"] = pd.to_numeric(df["motion_score"], errors="coerce")
    df["label_score"] = pd.to_numeric(df["label_score"], errors="coerce")

    df["max_overlap_score"] = df[["motion_score", "label_score"]].max(axis=1)

    df = df.sort_values(
        by=["max_overlap_score", "ntu_id"],
        ascending=[True, True],
    )

    df["summary"] = df.apply(
        lambda r: (
            f"A{int(r['ntu_id'])}: {r['ntu_label']} | "
            f"motion={r['motion_score']:.4f} ({r['motion_group']}), "
            f"label={r['label_score']:.4f} ({r['label_group']}), "
            f"max={r['max_overlap_score']:.4f}"
        ),
        axis=1,
    )

    out = df[["summary"]]
    out.to_csv(args.output_csv, index=False)

    print("Saved:", args.output_csv)
    print("Rows written:", len(out))


if __name__ == "__main__":
    main()
