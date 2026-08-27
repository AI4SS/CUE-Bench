"""Public evaluation config for CUE-Bench.

This file intentionally reads API keys from environment variables only.
Set the corresponding environment variable before running `llm_evaluate.py`.
"""

import os


MODELS = {
    "deepseek-v4-flash": {
        "api_key_var": "DEEPSEEK_API_KEY",
        "base_url_var": "DEEPSEEK_BASE_URL",
        "model_name": os.environ.get("DEEPSEEK_MODEL", "deepseek-ai/DeepSeek-V4-Flash"),
    },
    "gpt-4o-mini": {
        "api_key_var": "OPENAI_API_KEY",
        "base_url_var": "OPENAI_BASE_URL",
        "model_name": os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
    },
    "llama-4-maverick": {
        "api_key_var": "LLAMA_API_KEY",
        "base_url_var": "LLAMA_BASE_URL",
        "model_name": os.environ.get("LLAMA_MODEL", "meta-llama/llama-4-maverick"),
    },
    "llama-3.1-8b-instruct": {
        "api_key_var": "LLAMA31_API_KEY",
        "base_url_var": "LLAMA31_BASE_URL",
        "model_name": os.environ.get("LLAMA31_MODEL", "meta-llama/llama-3.1-8b-instruct"),
    },
    "qwen3-8b": {
        "api_key_var": "QWEN_API_KEY",
        "base_url_var": "QWEN_BASE_URL",
        "model_name": os.environ.get("QWEN_MODEL", "qwen/qwen3-8b"),
    },
    "glm-5.1": {
        "api_key_var": "GLM_API_KEY",
        "base_url_var": "GLM_BASE_URL",
        "model_name": os.environ.get("GLM_MODEL", "Pro/zai-org/GLM-5.1"),
        "disable_thinking": "enable_thinking_false",
        "force_disable_thinking": True,
    },
}


DEFAULT_BASE_URL = os.environ.get("OPENAI_COMPATIBLE_BASE_URL", "https://api.openai.com/v1")

METHODS = [
    "direct",
    "cot",
    "chain",
    "structure_only",
    "gold_p",
    "gold_pi",
]

TASKS = {
    "stance": {
        "label_field": "affective_stance",
        "labels": [
            "Positive", "Formulaic Positive", "Sarcastic Negative",
            "Understated Positive", "Neutral", "Veiled Negative",
            "Affiliative Positive", "Reportive Negative", "Negative",
        ],
        "prompt_files": {
            "direct": "direct_stance.txt",
            "cot": "cot_stance.txt",
            "chain": "chain_stance.txt",
        },
        "output_key": {
            "direct": "affective_stance",
            "cot": "affective_stance",
            "chain": "affective_stance",
        },
    },
    "intent": {
        "label_field": "pragmatic_intent",
        "labels": [
            "Authenticity", "Politeness", "Suppression", "Irony",
            "Resistance", "Functional", "Humor", "Empathy",
        ],
        "prompt_files": {
            "direct": "direct_intent.txt",
            "cot": "cot_intent.txt",
            "chain": "chain_intent.txt",
            "structure_only": "structure_only_intent.txt",
            "gold_p": "gold_stance_to_intent.txt",
        },
        "output_key": {
            "direct": "pragmatic_intent",
            "cot": "pragmatic_intent",
            "chain": "pragmatic_intent",
            "structure_only": "pragmatic_intent",
            "gold_p": "pragmatic_intent",
        },
    },
    "emotion": {
        "label_field": "fine_grained_emotion",
        "labels": [
            "Love", "Delight", "Pride", "Optimism", "Hope", "Curiosity",
            "Submission", "Awe", "Anxiety",
            "Despair", "Pessimism", "Shame", "Guilt", "Remorse", "Disappointment",
            "Outrage", "Aggression", "Contempt", "Dominance",
            "Envy", "Cynicism", "Morbidness", "Unbelief",
            "Sentimentality", "Neutral",
        ],
        "prompt_files": {
            "direct": "direct_emotion.txt",
            "cot": "cot_emotion.txt",
            "chain": "chain_emotion.txt",
            "structure_only": "structure_only_emotion.txt",
            "gold_p": "gold_stance_to_emotion.txt",
            "gold_pi": "gold_stance_intent_to_emotion.txt",
        },
        "output_key": {
            "direct": "fine_grained_emotion",
            "cot": "fine_grained_emotion",
            "chain": "fine_grained_emotion",
            "structure_only": "fine_grained_emotion",
            "gold_p": "fine_grained_emotion",
            "gold_pi": "fine_grained_emotion",
        },
    },
}

TEMPERATURE = 0.0
MAX_TOKENS = 1024
MAX_RETRIES = 3
RETRY_DELAY = 2
FEW_SHOT_N = {
    "stance": 9,
    "intent": 8,
    "emotion": 5,
}
