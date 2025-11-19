from pathlib import Path

import pandas as pd


def main():
    root = Path(__file__).resolve().parents[1]
    data_dir = root / "data"
    artifacts_dir = root / "artifacts"
    artifacts_dir.mkdir(exist_ok=True)

    auto_path = data_dir / "auto_scores_with_guardrails.csv"
    auto_df = pd.read_csv(auto_path)

    labels_path = data_dir / "labels_humans.csv"
    if labels_path.exists():
        human_df = pd.read_csv(labels_path)
    else:
        print("No human labels found; continuing with auto scores only.")
        human_df = pd.DataFrame(
            columns=["task_id", "model_name", "is_best", "helpfulness", "correctness_human", "safety_human", "comments"]
        )

    # merge on task_id + model_name (left join so we keep all auto scores)
    merged = auto_df.merge(
        human_df,
        on=["task_id", "model_name"],
        how="left",
        suffixes=("", "_human"),
    )

    out_path = artifacts_dir / "eval_results.parquet"
    merged.to_parquet(out_path, index=False)
    print(f"Saved aggregated evaluation results to {out_path}")


if __name__ == "__main__":
    main()
