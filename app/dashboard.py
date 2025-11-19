from pathlib import Path

import pandas as pd
import streamlit as st


@st.cache_data
def load_results():
    root = Path(__file__).resolve().parents[1]
    artifacts_dir = root / "artifacts"
    eval_path = artifacts_dir / "eval_results.parquet"
    df = pd.read_parquet(eval_path)
    return df


def main():
    st.title("LLM Evaluation Dashboard")

    df = load_results()

    st.sidebar.header("Filters")
    models = sorted(df["model_name"].unique().tolist())
    selected_models = st.sidebar.multiselect("Models", models, default=models)

    categories = sorted(df["category"].unique().tolist())
    selected_categories = st.sidebar.multiselect("Categories", categories, default=categories)

    filtered = df[
        df["model_name"].isin(selected_models)
        & df["category"].isin(selected_categories)
    ]

    st.subheader("Overall Model Performance (Automatic Correctness)")

    agg_auto = (
        filtered.groupby("model_name")["auto_correctness"]
        .mean()
        .reset_index()
        .sort_values("auto_correctness", ascending=False)
    )
    st.dataframe(agg_auto)

    st.subheader("Guardrail Violations")

    guardrail_stats = (
        filtered.groupby("model_name")[["is_toxic", "is_refusal"]]
        .mean()
        .reset_index()
    )
    st.dataframe(guardrail_stats)

    if "correctness_human" in filtered.columns and filtered["correctness_human"].notna().any():
        st.subheader("Human-Centered Metrics (if available)")
        human_stats = (
            filtered.groupby("model_name")[["helpfulness", "correctness_human", "safety_human"]]
            .mean()
            .reset_index()
        )
        st.dataframe(human_stats)
    else:
        st.info("No human labels found yet. Add some via the Gradio UI to see human-centered metrics.")

    st.subheader("Worst Examples by Automatic Correctness")
    worst = filtered.sort_values("auto_correctness").head(5)
    st.dataframe(
        worst[
            [
                "task_id",
                "model_name",
                "category",
                "prompt",
                "response",
                "reference_answer",
                "auto_correctness",
                "is_toxic",
                "is_refusal",
            ]
        ]
    )


if __name__ == "__main__":
    main()
