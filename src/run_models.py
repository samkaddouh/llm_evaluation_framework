import random
from pathlib import Path

import numpy as np
import pandas as pd


RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


def dummy_model_reasoning(prompt: str, reference: str, model_quality: float) -> str:
    """
    Simulate a reasoning model by either giving the correct numeric answer
    or a slightly wrong one depending on model_quality (0–1).
    """
    if random.random() < model_quality:
        return reference

    # produce a wrong but plausible numeric answer
    try:
        ref_val = float(reference)
        noise = np.random.normal(loc=1.0, scale=1.5)
        wrong_val = ref_val + noise
        return str(round(wrong_val, 2))
    except ValueError:
        return "I'm not sure, but it seems to be around " + reference


def dummy_model_summarization(source_summary: str, model_quality: float) -> str:
    """
    Simulate summarization: good models stay close to reference, weaker ones
    add noise or unnecessary fluff.
    """
    if random.random() < model_quality:
        return source_summary

    # add some fluff / slightly distort
    return source_summary + " Overall, this has various implications for stakeholders."


def dummy_model_sentiment(reference_label: str, model_quality: float) -> str:
    """
    Simulate sentiment classification with a given accuracy.
    """
    if random.random() < model_quality:
        return reference_label

    labels = ["positive", "negative", "neutral"]
    other_labels = [l for l in labels if l != reference_label]
    return random.choice(other_labels)


def generate_outputs_for_model(tasks: pd.DataFrame, model_name: str, quality: float) -> pd.DataFrame:
    rows = []

    for _, row in tasks.iterrows():
        task_id = row["task_id"]
        category = row["category"]
        prompt = row["prompt"]
        reference = row["reference_answer"]

        if category == "math_reasoning":
            response = dummy_model_reasoning(prompt, reference, quality)
        elif category == "summarization":
            response = dummy_model_summarization(reference, quality)
        elif category == "sentiment_classification":
            response = dummy_model_sentiment(reference, quality)
        else:
            response = "I am not configured for this task type."

        rows.append(
            {
                "task_id": task_id,
                "model_name": model_name,
                "response": response,
            }
        )

    return pd.DataFrame(rows)


def main():
    root = Path(__file__).resolve().parents[1]
    data_dir = root / "data"
    outputs_dir = data_dir / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)

    tasks = pd.read_csv(data_dir / "tasks.csv")

    # Simulate three models with different "quality" levels
    configs = [
        ("llama3_dummy", 0.8),
        ("mistral_dummy", 0.7),
        ("gpt4_dummy", 0.9),
    ]

    for model_name, quality in configs:
        df = generate_outputs_for_model(tasks, model_name, quality)
        out_path = outputs_dir / f"{model_name}_outputs.csv"
        df.to_csv(out_path, index=False)
        print(f"Saved outputs for {model_name} to {out_path}")


if __name__ == "__main__":
    main()
