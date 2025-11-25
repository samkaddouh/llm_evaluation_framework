

import random

# --- Dummy model logic for demo ---------------------------------

def _gpt4_dummy(prompt: str) -> str:
    return f"[GPT-4 Dummy] Answer: {prompt[::-1]}"

def _llama3_dummy(prompt: str) -> str:
    return f"[LLaMA-3 Dummy] Answer: {prompt.upper()}"

def _mistral_dummy(prompt: str) -> str:
    return f"[Mistral Dummy] Answer: {''.join(random.sample(prompt, len(prompt)))}"


# --- PUBLIC API -------------------------------------------------

def generate_response(model_name: str, prompt: str) -> str:
    """
    Universal generation function used by Streamlit.
    """

    model_name = model_name.lower()

    if model_name == "gpt4_dummy":
        return _gpt4_dummy(prompt)

    elif model_name == "llama3_dummy":
        return _llama3_dummy(prompt)

    elif model_name == "mistral_dummy":
        return _mistral_dummy(prompt)

    else:
        return "Unknown model. Available: gpt4_dummy, llama3_dummy, mistral_dummy."
