from pathlib import Path
import random

import gradio as gr
import pandas as pd


RANDOM_SEED = 123
random.seed(RANDOM_SEED)


def load_data():
    root = Path(__file__).resolve().parents[1]
    data_dir = root / "data"
    scored_path = data_dir / "auto_scores_with_guardrails.csv"
    df = pd.read_csv(scored_path)

    # sort for reproducibility
    df = df.sort_values(["task_id", "model_name"]).reset_index(drop=True)
    return df, data_dir


def sample_task(df: pd.DataFrame):
    task_id = random.choice(df["task_id"].unique().tolist())
    subset = df[df["task_id"] == task_id]

    prompt = subset["prompt"].iloc[0]
    category = subset["category"].iloc[0]

    models = subset["model_name"].tolist()
    responses = subset["response"].tolist()

    return task_id, category, prompt, models, responses


DF, DATA_DIR = load_data()


def next_example():
    task_id, category, prompt, models, responses = sample_task(DF)
    # Build a combined display string
    text = f"Task ID: {task_id}\nCategory: {category}\n\nPrompt:\n{prompt}\n\n"
    for m, r in zip(models, responses):
        text += f"---\nModel: {m}\nResponse:\n{r}\n\n"
    return task_id, text, models


def save_feedback(task_id, models, best_model, helpfulness, correctness, safety, comments):
    labels_path = DATA_DIR / "labels_humans.csv"

    records = []
    for m in models:
        records.append(
            {
                "task_id": task_id,
                "model_name": m,
                "is_best": int(m == best_model),
                "helpfulness": helpfulness,
                "correctness_human": correctness,
                "safety_human": safety,
                "comments": comments,
            }
        )

    df_new = pd.DataFrame(records)

    if labels_path.exists():
        df_existing = pd.read_csv(labels_path)
        df_all = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        df_all = df_new

    df_all.to_csv(labels_path, index=False)
    return "Feedback saved! Click 'Next example' to annotate another sample."


with gr.Blocks() as demo:
    gr.Markdown("# LLM Evaluation – Human Annotation UI")

    task_id_state = gr.State("")
    models_state = gr.State([])

    with gr.Row():
        next_btn = gr.Button("Next example")

    with gr.Row():
        task_display = gr.Textbox(label="Task & Model Responses", lines=20)

    with gr.Row():
        best_model = gr.Textbox(label="Which model performed best? (enter model name)")
    with gr.Row():
        helpfulness = gr.Slider(0, 5, step=1, value=3, label="Helpfulness (0–5)")
        correctness = gr.Slider(0, 5, step=1, value=3, label="Correctness (0–5)")
        safety = gr.Slider(0, 5, step=1, value=4, label="Safety (0–5)")
    comments = gr.Textbox(label="Comments", lines=3)

    save_btn = gr.Button("Save feedback")
    status = gr.Markdown("")

    def on_next():
        t_id, text, models = next_example()
        return t_id, models, text

    next_btn.click(on_next, outputs=[task_id_state, models_state, task_display])

    def on_save(task_id, models, best_model, helpfulness, correctness, safety, comments):
        return save_feedback(task_id, models, best_model, helpfulness, correctness, safety, comments)

    save_btn.click(
        on_save,
        inputs=[task_id_state, models_state, best_model, helpfulness, correctness, safety, comments],
        outputs=status,
    )


if __name__ == "__main__":
    demo.launch()
