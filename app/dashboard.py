from pathlib import Path
import sys

import pandas as pd
import streamlit as st


# -------------------------------------------------------
# Data loader
# -------------------------------------------------------
@st.cache_data
def load_results():
    root = Path(__file__).resolve().parents[1]
    artifacts_dir = root / "artifacts"
    eval_path = artifacts_dir / "eval_results.parquet"

    if not eval_path.exists():
        raise FileNotFoundError(
            f"Could not find {eval_path}. Make sure you've run the evaluation pipeline "
            "and generated eval_results.parquet."
        )

    return pd.read_parquet(eval_path)


# -------------------------------------------------------
# Main app
# -------------------------------------------------------
def main():
    st.title("LLM Evaluation Dashboard")

    # Load evaluation results
    df = load_results()

    # -------------------------------
    # Sidebar Filters
    # -------------------------------
    st.sidebar.header("Filters")

    models = sorted(df["model_name"].unique().tolist())
    selected_models = st.sidebar.multiselect("Models", models, default=models)

    categories = sorted(df["category"].unique().tolist())
    selected_categories = st.sidebar.multiselect("Categories", categories, default=categories)

    filtered = df[
        df["model_name"].isin(selected_models)
        & df["category"].isin(selected_categories)
    ]

    # -------------------------------
    # Overall Auto Correctness
    # -------------------------------
    st.subheader("Overall Model Performance (Automatic Correctness)")

    agg_auto = (
        filtered.groupby("model_name")["auto_correctness"]
        .mean()
        .reset_index()
        .sort_values("auto_correctness", ascending=False)
    )
    st.dataframe(agg_auto, use_container_width=True)

    # -------------------------------
    # Guardrail Violations
    # -------------------------------
    st.subheader("Guardrail Violations")

    guardrail_stats = (
        filtered.groupby("model_name")[["is_toxic", "is_refusal"]]
        .mean()
        .reset_index()
    )
    st.dataframe(guardrail_stats, use_container_width=True)

    # -------------------------------
    # Human Centered Metrics (optional)
    # -------------------------------
    if "correctness_human" in filtered.columns and filtered["correctness_human"].notna().any():
        st.subheader("Human-Centered Metrics (if available)")

        human_stats = (
            filtered.groupby("model_name")[["helpfulness", "correctness_human", "safety_human"]]
            .mean()
            .reset_index()
        )
        st.dataframe(human_stats, use_container_width=True)
    else:
        st.info(
            "No human labels found yet. Add some via the Gradio UI to see human-centered metrics."
        )

    # -------------------------------
    # Worst Examples
    # -------------------------------
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
        ],
        use_container_width=True,
    )

    # =======================================================
    # LLM Playground (Interactive Chat)
    # =======================================================
    st.divider()
    st.header("🗣️ LLM Playground")

    # Ensure project root is on sys.path so `src.*` imports work under Streamlit
    ROOT = Path(__file__).resolve().parents[1]
    if str(ROOT) not in sys.path:
        sys.path.append(str(ROOT))

    # Import AFTER sys.path adjustment
    from src.chat_models import generate_response

    model_choice = st.selectbox(
        "Choose a model",
        options=["llama3_dummy", "mistral_dummy", "gpt4_dummy"],
        index=2,
    )

    user_prompt = st.text_area(
        "Enter your question / prompt",
        placeholder="Ask anything your models should answer...",
        height=120,
    )

    if st.button("Generate Answer"):
        if not user_prompt.strip():
            st.warning("Please enter a prompt.")
        else:
            with st.spinner(f"Generating response with {model_choice}..."):
                answer = generate_response(model_choice, user_prompt)

            st.subheader("Model Output")
            st.write(answer)


if __name__ == "__main__":
    main()
