import pandas as pd
from pathlib import Path


def build_tasks() -> pd.DataFrame:
    """Create a small synthetic task set for LLM evaluation."""

    tasks = []

    # 1) Reasoning / math word problems with numeric answers
    reasoning_prompts = [
        (
            "math_reasoning",
            "John has 3 apples and buys 5 more. How many apples does he have now?",
            "8"
        ),
        (
            "math_reasoning",
            "A train leaves at 3 PM and arrives at 6:30 PM. How many hours is the journey?",
            "3.5"
        ),
        (
            "math_reasoning",
            "Sarah had 12 cookies and gave 7 to her friend. How many cookies does she have left?",
            "5"
        ),
    ]

    for i, (category, prompt, answer) in enumerate(reasoning_prompts, start=1):
        tasks.append(
            {
                "task_id": f"r{i}",
                "category": category,
                "prompt": prompt,
                "reference_answer": answer,
            }
        )

    # 2) Summarization tasks
    summarization_sources = [
        (
            "summarization",
            "The company reported a 20% increase in quarterly revenue driven by strong demand "
            "for its cloud services. However, rising operating costs slightly reduced overall profit margins.",
            "Company revenue increased by 20% due to cloud demand, but higher costs reduced profit margins."
        ),
        (
            "summarization",
            "A new city-wide bike-sharing program launched last month and has already attracted over "
            "10,000 registered users. Officials hope it will reduce traffic congestion and promote sustainability.",
            "A new bike-sharing program gained 10,000 users in a month and aims to cut traffic and support sustainability."
        ),
    ]

    for i, (category, source, reference_summary) in enumerate(summarization_sources, start=1):
        tasks.append(
            {
                "task_id": f"s{i}",
                "category": category,
                "prompt": f"Summarize the following text in 1–2 sentences:\n\n{source}",
                "reference_answer": reference_summary,
            }
        )

    # 3) Classification tasks (sentiment)
    classification_prompts = [
        (
            "sentiment_classification",
            "The product quality is amazing and I would definitely buy it again.",
            "positive"
        ),
        (
            "sentiment_classification",
            "The service was terrible and I will not recommend this to anyone.",
            "negative"
        ),
        (
            "sentiment_classification",
            "The movie was okay, not great but not terrible either.",
            "neutral"
        ),
    ]

    for i, (category, text, label) in enumerate(classification_prompts, start=1):
        tasks.append(
            {
                "task_id": f"c{i}",
                "category": category,
                "prompt": f"Classify the sentiment of the following text as positive, negative, or neutral:\n\n{text}",
                "reference_answer": label,
            }
        )

    return pd.DataFrame(tasks)


def main():
    root = Path(__file__).resolve().parents[1]
    data_dir = root / "data"
    data_dir.mkdir(exist_ok=True)

    tasks_df = build_tasks()
    out_path = data_dir / "tasks.csv"
    tasks_df.to_csv(out_path, index=False)
    print(f"Saved {len(tasks_df)} tasks to {out_path}")


if __name__ == "__main__":
    main()
