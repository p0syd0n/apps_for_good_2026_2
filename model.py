from sentence_transformers import SentenceTransformer, CrossEncoder
import numpy as np
import torch
import os
from dotenv import load_dotenv
from google import genai

load_dotenv() 

def embed(phrase):
    result = client.models.embed_content(
        model="gemini-embedding-001",
        contents="What is the meaning of life?"
    )
    return result.embeddings

def load_model():
    # Use SentenceTransformer to load BGE-M3. It's the same model, 
    # but uses a more stable loading pathway.
    sim_model = SentenceTransformer('BAAI/bge-m3')
    nli_model = CrossEncoder('cross-encoder/nli-deberta-v3-base')
    print("Models Loaded Successfully")
    return nli_model, sim_model

def inference(models, phrase, relevant_paragraph):
    (nli_model, sim_model) = models
    
    # 1. Generate Embeddings using SentenceTransformer
    phrase_vec = sim_model.encode([phrase])
    paper_vec = sim_model.encode([relevant_paragraph])

    # 2. Calculate Cosine Similarity
    # sentence-transformers outputs are usually normalized, so dot product works
    similarity = np.dot(phrase_vec[0], paper_vec[0])
    print(f"\nTopic Similarity Score: {similarity:.4f}")

    if similarity < 0.6: # Adjusted threshold slightly
        print("Irrelevant.")
        #print("Verdict: Neutral (The paper doesn't talk about this topic.)")
        return

    #print("Topic is relevant. Proceeding to logic check...")
        
    # 4. Logical Inference
    logits = nli_model.predict([(phrase, relevant_paragraph)])
    probabilities = torch.nn.functional.softmax(torch.tensor(logits), dim=1).numpy()[0]

    # Deberta NLI Labels: 0: Contradiction, 1: Neutral, 2: Entailment
    labels = ['contradicts', 'ftg fdp', 'entails']

    # for label, prob in zip(labels, probabilities):
    #     print(f"{label}: {prob:.2%}")

    # 5. Final Verdict Logic
    max_prob = np.max(probabilities)
    if max_prob < 0.60:
        verdict = "Neutral (Inconclusive/Mixed Evidence)"
    else:
        verdict = labels[np.argmax(probabilities)]

    print(f"The phrase '{phrase}' {verdict} '{relevant_paragraph}'.")

def split_to_atoms(text):
    response = client.models.generate_content(
        model="gemini-2.5-flash", contents=f"{os.getenv("ATOM_SPLIT_PROMPT")}\n\n{text}"
    )
    return [claim.replace("\n", "") for claim in response.text.split(".") if len(claim)!=0]

client = genai.Client()

models = load_model()

paragraph = """Hydrogels do not have dynamic cues. Hydrogels do not have structural complexity. These factors limit their function."""

while True:
    user_phrase = input("\nEnter phrase to test: ")
    if user_phrase.lower() in ['exit', 'quit']: break
    user_claims = split_to_atoms(user_phrase)
    paper_claims = split_to_atoms(paragraph)

    for user_claim in user_claims:
        for paper_claim in paper_claims:
            inference(models, user_claim, paper_claim)