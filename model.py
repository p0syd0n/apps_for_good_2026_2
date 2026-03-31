from sentence_transformers import SentenceTransformer, CrossEncoder
import numpy as np
import torch
import os
from dotenv import load_dotenv
from google import genai

load_dotenv() 

def embed(client, phrase):
    result = client.models.embed_content(
        model="gemini-embedding-001",
        contents=phrase
    )
    return result.embeddings[0].values

def get_llm():
    return genai.Client()

def use_llm(client, content):
    return client.models.generate_content(
        model="gemini-2.5-flash", contents=content
    )
    
def load_model():
    sim_model = SentenceTransformer('BAAI/bge-m3')
    nli_model = CrossEncoder('cross-encoder/nli-deberta-v3-base')
    print("Models Loaded Successfully")
    return nli_model, sim_model

def similarity(client, models, phrase, phrase1):
    (_, sim_model) = models
    
    phrase_vec = embed(client, phrase)
    paper_vec = embed(client, phrase1)

    similarity = np.dot(phrase_vec, paper_vec)
    print(f"\nTopic Similarity Score: {similarity:.4f}")

    return similarity 


def inference(client, models, phrase, relevant_paragraph):
    (nli_model, sim_model) = models
    
    # 1. Generate Embeddings using SentenceTransformer
    phrase_vec = embed(client, phrase)
    paper_vec = embed(client, relevant_paragraph)

    # 2.  dot product for simularity
    similarity = np.dot(phrase_vec, paper_vec)
    print(f"\nTopic Similarity Score: {similarity:.4f}")

    if similarity < 0.6:
        #print("Verdict: Neutral (The paper doesn't talk about this topic.)")
        return -1

    # NLI 
    logits = nli_model.predict([(phrase, relevant_paragraph)])
    probabilities = torch.nn.functional.softmax(torch.tensor(logits), dim=1).numpy()[0]

    # 0: Contradiction, 1: Neutral, 2: Entailment
    labels = ['contradicts', 'fneutral', 'entails']

    # for label, prob in zip(labels, probabilities):
    #     print(f"{label}: {prob:.2%}")

    # 5. Final Verdict Logic
    max_prob = np.max(probabilities)
    if max_prob < 0.60:
        verdict = 1
    else:
        verdict = np.argmax(probabilities)



    print(f"The phrase '{phrase}' {verdict} '{relevant_paragraph}'.")
    return np.argmax(probabilities)

def split_to_atoms(client, text):
    response = client.models.generate_content(
        model="gemini-2.5-flash", contents=f"{os.getenv("ATOM_SPLIT_PROMPT")}\n\n{text}"
    )
    return [claim.replace("\n", "") for claim in response.text.split(".") if len(claim)!=0]

def get_abstract_atoms(client, text):
    response = client.models.generate_content(
        model="gemini-2.5-flash", contents=f"{os.getenv("GET_ABSTRACT_ATOMS_PROMPT")}\n\n{text}"
    )
    return [claim.replace("\n", "") for claim in response.text.split(".") if len(claim)!=0]

def extract_keywords(client, text):
    response = client.models.generate_content(
        model="gemini-2.5-flash", contents=f"{os.getenv("KEYWORDS_PROMPT")}\n\n{text}"
    )
    print(response.text)
    return response.text
    #return [claim.replace("\n", "") for claim in response.text.split(".") if len(claim)!=0]

def main():

    client = get_llm()

    models = load_model()

    paragraph = """Hydrogels do not have dynamic cues. Hydrogels do not have structural complexity. These factors limit their function."""

    while True:
        user_phrase = input("\nEnter phrase to test: ")
        if user_phrase.lower() in ['exit', 'quit']: break
        user_claims = split_to_atoms(client, user_phrase)
        paper_claims = split_to_atoms(client, paragraph)

        results = []
        for user_claim in user_claims:
            for paper_claim in paper_claims:
                result = inference(client, models, user_claim, paper_claim)
                results.append(result)

if __name__ == "__main__":
    main()

