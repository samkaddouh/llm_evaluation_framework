from pathlib import Path

import numpy as np
import pandas as pd


def score_math_reasoning(pred: str, ref: str) -> float:
    try:
        pred_val = float(pred.strip())
        ref_val = float(ref.strip())
        # full credit if within small tolerance
        if abs(pred_val - ref_val) < 1e-3:
            return 1.0
        # partial credit if somewhat close
        diff = abs(pred_val - ref_val)
        return max(0.0, 1.0 - diff / max(1.0, abs(ref_val)))
    except Exception:
        return 0.0


def score_sentiment(pred: str, ref: str) -> float:
    pred = (pred or "").strip().lower()
    ref = (ref or "").strip().lower()
    return 1.0 if pred == ref else 0.0


def simple_overlap_score(pred: str, ref: str) -> float:
    """Very simple token overlap score between prediction and reference."""
    pred_tokens = set(pred.lower().split())
    ref_tokens = set(ref.lower().split())
    if not ref_tokens:
        return 0.0
    overlap = pred_tokens.intersection(ref_tokens)
    return len(overlap) / len(ref_tokens)


def apply_rubric(tasks: pd.DataFrame, outputs: pd.DataFrame) -> pd.DataFrame:
    merged = outputs.merge(tasks, on="task_id", how="left")
    scores = []

    for _, row in merged.iterrows():
        category = row["category"]
        pred = str(row["response"])
        ref = str(row["reference_answer"])

        if category == "math_reasoning":
            correctness = score_math_reasoning(pred, ref)
        elif category == "sentiment_classification":
            correctness = score_sentiment(pred, ref)
        elif category == "summarization":
            correctness = simple_overlap_score(pred, ref)
        else:
            correctness = 0.0

        scores.append(correctness)

    merged["auto_correctness"] = scores
    return merged


def main():
    root = Path(__file__).resolve().parents[1]
    data_dir = root / "data"
    outputs_dir = data_dir / "outputs"

    tasks = pd.read_csv(data_dir / "tasks.csv")

    # Load all model outputs and score them
    all_scored = []

    for csv_path in outputs_dir.glob("*_outputs.csv"):
        outputs = pd.read_csv(csv_path)
        scored = apply_rubric(tasks, outputs)
        all_scored.append(scored)

    if not all_scored:
        print("No outputs found in data/outputs")
        return

    result = pd.concat(all_scored, ignore_index=True)
    out_path = data_dir / "auto_scores.csv"
    result.to_csv(out_path, index=False)
    print(f"Saved auto-scored results to {out_path}")


if __name__ == "__main__":
    main()
