"""
Fact-checking / NLI inference pipeline.
All LLM/embedding calls go through an LLMProvider tp swap providers freely.
"""

import numpy as np
import torch
import os
from sentence_transformers import CrossEncoder
from llm_providers import LLMProvider, get_provider



# Model loading (local NLI model only)

def load_nli_model() -> CrossEncoder:
    model = CrossEncoder("cross-encoder/nli-deberta-v3-base")
    print("NLI model loaded.")
    return model


# Embedding helpers (provider-agnostic)

def similarity(provider: LLMProvider, a: str, b: str) -> float:
    """Cosine-equivalent dot product between two embedded strings."""
    score = float(np.dot(provider.embed(a), provider.embed(b)))
    print(f"Similarity: {score:.4f}")
    return score


# NLI inference

LABELS = ["contradicts", "neutral", "entails"]
SIMILARITY_THRESHOLD = 0.60
CONFIDENCE_THRESHOLD = 0.60

#     Returns an index into LABELS: 0=contradicts, 1=neutral, 2=entails.
def run_nli(nli_model: CrossEncoder, premise: str, hypothesis: str) -> int:
    """
    Returns an index into LABELS: 0=contradicts, 1=neutral, 2=entails.
    """
    logits = nli_model.predict([(premise, hypothesis)])
    probs  = torch.nn.functional.softmax(torch.tensor(logits), dim=1).numpy()[0]
    return int(np.argmax(probs)) if np.max(probs) >= CONFIDENCE_THRESHOLD else 1

# make a NLI inference between two statements
def inference(
    provider:  LLMProvider,
    nli_model: CrossEncoder,
    claim:     str,
    context:   str,
) -> int:
    """
    Compare `claim` against `context`.

    Returns:
        -1  topic mismatch (similarity below threshold)
         0  contradicts
         1  neutral / uncertain
         2  entails
    """
    if similarity(provider, claim, context) < SIMILARITY_THRESHOLD:
        return -1

    verdict = run_nli(nli_model, claim, context)
    print(f"  [{LABELS[verdict]}]  '{claim}'  vs  '{context}'")
    return verdict


# Text splitting (provider-agnostic prompts)

def split_to_atoms(provider: LLMProvider, text: str) -> list[str]:
    prompt   = os.getenv("ATOM_SPLIT_PROMPT", "Split the following text into atomic factual claims, one per line:")
    response = provider.generate(prompt=text, system=prompt)
    return [c.replace("\n", "").strip() for c in response.split(".") if c.strip()]


def get_abstract_atoms(provider: LLMProvider, text: str) -> list[str]:
    prompt   = os.getenv("GET_ABSTRACT_ATOMS_PROMPT", "Extract the key abstract claims from this text, one per line:")
    response = provider.generate(prompt=text, system=prompt)
    return [c.replace("\n", "").strip() for c in response.split(".") if c.strip()]


def extract_keywords(provider: LLMProvider, text: str) -> str:
    prompt   = os.getenv("KEYWORDS_PROMPT", "Extract the key terms and concepts from this text as a comma-separated list:")
    response = provider.generate(prompt=text, system=prompt)
    print(response)
    return response


# tests

def main():
    provider  = get_provider("gemini")
    nli_model = load_nli_model()

    paragraph = (
        "Hydrogels do not have dynamic cues. "
        "Hydrogels do not have structural complexity. "
        "These factors limit their function."
    )

    while True:
        user_phrase = input("\nEnter phrase to test (or 'exit'): ").strip()
        if user_phrase.lower() in ("exit", "quit"):
            break

        user_claims  = split_to_atoms(provider, user_phrase)
        paper_claims = split_to_atoms(provider, paragraph)

        results = [
            inference(provider, nli_model, uc, pc)
            for uc in user_claims
            for pc in paper_claims
        ]
        print(f"\nResult vector: {results}")


if __name__ == "__main__":
    main()