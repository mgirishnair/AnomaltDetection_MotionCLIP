#!/usr/bin/env python
# coding: utf-8

import argparse
import pandas as pd


def aggregate_long_csv(path, score_name):
    df = pd.read_csv(path)

    required = [
        "ntu_id",
        "ntu_label",
        "babel_rank",
        "babel_label",
        "raw_similarity",
        "corrected_similarity",
        "overall_ntu_overlap_score",
        "overlap_group",
    ]

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing columns in {path}: {missing}\n"
            f"Available columns: {list(df.columns)}"
        )

    # One row per NTU class.
    grouped = (
        df.sort_values(["ntu_id", "babel_rank"])
        .groupby(["ntu_id", "ntu_label"], as_index=False)
        .agg(
            overlap_score=("overall_ntu_overlap_score", "first"),
            overlap_group=("overlap_group", "first"),
            top_babel_labels=("babel_label", lambda x: " | ".join(x.astype(str).tolist())),
            top_raw_scores=("raw_similarity", lambda x: " | ".join(f"{v:.6f}" for v in x)),
            top_corrected_scores=("corrected_similarity", lambda x: " | ".join(f"{v:.6f}" for v in x)),
            top1_babel_label=("babel_label", "first"),
            top1_raw_similarity=("raw_similarity", "first"),
            top1_corrected_similarity=("corrected_similarity", "first"),
        )
    )

    grouped = grouped.rename(
        columns={
            "overlap_score": f"{score_name}_score",
            "overlap_group": f"{score_name}_group",
            "top_babel_labels": f"{score_name}_top_babel_labels",
            "top_raw_scores": f"{score_name}_top_raw_scores",
            "top_corrected_scores": f"{score_name}_top_corrected_scores",
            "top1_babel_label": f"{score_name}_top1_babel_label",
            "top1_raw_similarity": f"{score_name}_top1_raw_similarity",
            "top1_corrected_similarity": f"{score_name}_top1_corrected_similarity",
        }
    )

    return grouped


def group_to_num(group):
    mapping = {
        "low": 0,
        "medium": 1,
        "high": 2,
    }
    return mapping.get(str(group).lower(), -1)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--motion_long_csv", required=True)
    parser.add_argument("--label_long_csv", required=True)
    parser.add_argument("--output_csv", required=True)

    parser.add_argument("--motion_score_threshold", type=float, default=0.45)
    parser.add_argument("--label_score_threshold", type=float, default=0.55)

    args = parser.parse_args()

    motion = aggregate_long_csv(args.motion_long_csv, "motion")
    label = aggregate_long_csv(args.label_long_csv, "label")

    merged = motion.merge(
        label,
        on=["ntu_id", "ntu_label"],
        how="outer",
        validate="one_to_one",
    )

    merged["motion_score"] = pd.to_numeric(merged["motion_score"], errors="coerce")
    merged["label_score"] = pd.to_numeric(merged["label_score"], errors="coerce")

    # Conservative combined score: a class is only as "outside BABEL" as its strongest overlap view.
    # Low max score = low in both motion and label space.
    merged["max_overlap_score"] = merged[["motion_score", "label_score"]].max(axis=1)

    # Average score is also useful, but max score is stricter.
    merged["mean_overlap_score"] = merged[["motion_score", "label_score"]].mean(axis=1)

    merged["motion_group_num"] = merged["motion_group"].apply(group_to_num)
    merged["label_group_num"] = merged["label_group"].apply(group_to_num)
    merged["max_group_num"] = merged[["motion_group_num", "label_group_num"]].max(axis=1)

    merged["is_low_babel_overlap"] = (
        (merged["motion_score"] < args.motion_score_threshold)
        & (merged["label_score"] < args.label_score_threshold)
    )

    merged["is_low_by_group"] = (
        (merged["motion_group"].astype(str).str.lower() == "low")
        & (merged["label_group"].astype(str).str.lower() == "low")
    )

    # Rank best candidates first.
    merged = merged.sort_values(
        by=[
            "is_low_babel_overlap",
            "is_low_by_group",
            "max_overlap_score",
            "mean_overlap_score",
        ],
        ascending=[False, False, True, True],
    )

    cols = [
        "ntu_id",
        "ntu_label",

        "motion_score",
        "motion_group",
        "motion_top_babel_labels",
        "motion_top_raw_scores",
        "motion_top_corrected_scores",

        "label_score",
        "label_group",
        "label_top_babel_labels",
        "label_top_raw_scores",
        "label_top_corrected_scores",

        "max_overlap_score",
        "mean_overlap_score",
        "is_low_babel_overlap",
        "is_low_by_group",
    ]

    merged = merged[cols]
    merged.to_csv(args.output_csv, index=False)

    print("Saved:", args.output_csv)

    print("\nBest low-BABEL-overlap NTU candidates")
    print("-------------------------------------")

    candidates = merged[merged["is_low_babel_overlap"]].head(30)

    if len(candidates) == 0:
        print("No classes passed the strict score thresholds.")
        print("Showing lowest max-overlap classes instead:\n")
        candidates = merged.head(30)

    for _, row in candidates.iterrows():
        print(
            f"A{int(row['ntu_id'])}: {row['ntu_label']} | "
            f"motion={row['motion_score']:.4f} ({row['motion_group']}), "
            f"label={row['label_score']:.4f} ({row['label_group']}), "
            f"max={row['max_overlap_score']:.4f}"
        )


if __name__ == "__main__":
    main()
