from flask import Flask, request, render_template
import model
import scholar

N_PAPERS = 5

app = Flask(__name__)

@app.route('/')
def main():
    print("Main hit")
    return render_template("index.html")


@app.route("/inference")
def inference():
    query = request.args["query"]
    keywords = model.extract_keywords(client, query)
    papers = scholar.get_papers(keywords)

    query_atoms = model.split_to_atoms(client, query)

    papers_to_use = []

    for paper in papers:
        for query_atom in query_atoms:
            similarity = model.similarity(client, models, query_atom, paper["title"])
            if similarity > 0.6:
                paper["similarity"] = similarity
                papers_to_use.append(paper)

    papers_to_use.sort(key=lambda d: d["similarity"], reverse=True)

    
    if len(papers_to_use) > N_PAPERS:
        paper = papers_to_use[:N_PAPERS]

    for paper in papers_to_use:
        print(paper["title"] + " : "+paper["link"])
        
    

    return "200"

client = model.genai.Client()

models = model.load_model()

app.run(host='0.0.0.0', port=8080, debug=True)