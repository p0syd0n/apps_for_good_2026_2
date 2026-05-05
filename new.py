import os
import json
import nltk
import requests
from flask import Flask, render_template
from flask_socketio import SocketIO, emit
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

# Setup NLTK
nltk.download('stopwords')
nltk.download('punkt_tab')  # Required for newer NLTK/Python versions
stop_words = set(stopwords.words('english'))

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

# API Keys
SEMANTIC_SCHOLAR_KEY = os.getenv("SEMANTIC_SCHOLAR_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

client = Groq(api_key=GROQ_API_KEY)

@app.route('/')
def mainer():
    return render_template("index.html")

def extract_keywords(query):
    words = word_tokenize(query)
    print("tokenized words")
    keywords = [w for w in words if w.lower() not in stop_words and w.isalnum()]
    return " ".join(keywords)

def get_papers(query, limit=5):
    url = f"https://api.semanticscholar.org/graph/v1/paper/search?query={query}&limit={limit}&fields=title,abstract"
    headers = {"x-api-key": SEMANTIC_SCHOLAR_KEY} if SEMANTIC_SCHOLAR_KEY else {}
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        return response.json().get('data', [])
    return []


@socketio.on('inference')
def handle_inference(data):
    query = data.get('query')
    if not query:
        emit('error', {'msg': 'No query provided'})
        return

    # Step 1: Keyword Extraction
    emit('progress', {'status': 'extracting_keywords', 'msg': 'Extracting keywords...'})
    keywords = extract_keywords(query)
    
    # Step 2: Semantic Scholar Search
    emit('progress', {'status': 'fetching_papers', 'msg': f'Searching Semantic Scholar for: {keywords}'})
    papers = get_papers(keywords)
    abstracts_text = "\n\n".join([f"Title: {p['title']}\nAbstract: {p.get('abstract', 'No abstract available')}" for p in papers])

    # Step 3: Groq LLM Inference
    emit('progress', {'status': 'llm_inference', 'msg': 'Analyzing papers with Llama 3.1 8B...'})
    
    prompt = f"""
    You are a research assistant. Based on the following research abstracts, determine if the statement "{query}" is true, false, or inconclusive.
    
    Abstracts:
    {abstracts_text}
    
    Return your answer strictly in the following JSON format:
    {{
        "verdict": "True/False/Inconclusive",
        "confidence": 0.0-1.0,
        "explanation": "Short summary of why"
    }}
    """

    try:
        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.1-8b-instant",
            response_format={"type": "json_object"}
        )
        
        result = json.loads(chat_completion.choices[0].message.content)
        
        # Step 4: Final Response
        emit('final_result', {
            'analysis': result,
            'papers': papers
        })
        
    except Exception as e:
        emit('error', {'msg': str(e)})

if __name__ == '__main__':
    socketio.run(app, debug=True, port=5000)