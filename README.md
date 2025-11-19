# LLM Evaluation & Guardrail Framework

    This project implements a complete, modular evaluation system for comparing multiple LLMs, similar to internal frameworks used at OpenAI, Anthropic, and large AI product teams. It supports:

- Task generation
- Model output generation
- Correctness scoring
- Automatic guardrails
- Human-in-the-loop evaluation
- Unified dashboard<br>

The goal is to demonstrate LLM evaluation, safety, prompt engineering, data pipelines, and visualization end-to-end.

# Features

1. Task Generation
Creates evaluation tasks across categories:
- math reasoning
- summarization
- sentiment classification
- open-ended reasoning<br>

Saved to: data/tasks.csv

2. Model Output Pipeline
Simulates outputs for multiple models:
- GPT-4 dummy
- Mistral dummy
- Llama-3 dummy<br>

(Additional models can be added easily)

```Outputs saved to data/outputs/.```

3. Automatic Scoring
Includes a rubric for:
- string-based correctness
- numeric answer matching
- heuristic scoring for summarization
- yes/no classification<br>

4. Guardrail Framework
Basic safety rules:
- toxicity
- refusal detection
- profanity
(Extensible for privacy, hallucination, safety levels)<br>

5. Human Annotation UI
A full Gradio annotation tool where a human evaluator scores:
- correctness
- quality (1–5)
- model preference<br>

```Stored in data/labels_humans.csv.```

6. Aggregation Pipeline
Merges:
- auto scores
- guardrails
- model outputs
- human labels<br>

Final evaluation saved as:
```artifacts/eval_results.parquet```

7. Streamlit Dashboard
Features:
- model correctness comparison
- category filtering
- toxic/refusal guardrail violations
- worst examples
- (optional) human metrics section
- per-model breakdown<br>





## How to Run
1. Set up environment
```python3 -m venv .venv```
```source .venv/bin/activate```
```pip install -r requirements.txt```

2. Generate evaluation tasks
```python src/generate_tasks.py```
3. Generate model outputs
```python src/run_models.py```
4. Score outputs automatically
```python src/scoring_rubric.py```
5. Run guardrails
```python src/guardrails.py```
6. (Optional) Add human labels
```python src/human_annotation_ui.py```
7. Aggregate everything
```python src/aggregate_results.py```
8. Launch dashboard
```streamlit run app/dashboard.py```


# What This Project Demonstrates

- Advanced LLM evaluation
- Prompt engineering
- Safety/guardrail design
- Human-in-the-loop evaluation workflows
- Building pipelines with modularity
- Data engineering + ML + LLM reasoning
- Dashboarding for stakeholders
- Understanding of how AI labs internally evaluate models