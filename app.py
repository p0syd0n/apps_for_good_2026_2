from flask import Flask, request, render_template
import model
import scholar
import requests
import re

N_PAPERS = 5

app = Flask(__name__)

@app.route('/')
def main():
    print("Main hit")
    return render_template("index.html")


@app.route("/inference")
def inference():
    query = request.args["query"]
    # Get keywords from a query
    keywords = model.extract_keywords(client, query)
    # What papers come up on this query
    papers = scholar.get_papers(keywords)

    # Split the query to individual atoms
    query_atoms = model.split_to_atoms(client, query)

    # Create the list of papers which are relevant (query atom vs paper title)
    # Sorted by similarity, ascending
    papers_to_use = []

    for paper in papers:
        for query_atom in query_atoms:
            similarity = model.similarity(client, models, query_atom, paper["title"])
            if similarity > 0.6:
                paper["similarity"] = similarity
                papers_to_use.append(paper)


    papers_to_use.sort(key=lambda d: d["similarity"], reverse=True)


    # Cut to max number of papers
    if len(papers_to_use) > N_PAPERS:
        paper = papers_to_use[:N_PAPERS]


    files = []
    for paper in papers_to_use:
        print(paper["title"] + " : " + paper["link"])
        with open("papers/" + paper['title'], "w") as file:
            try:
                file.write(requests.get(paper["link"]).content.decode('utf-8'))
                files.append(paper)
            except Exception as e:
                print("Problem with request/write: " + str(e))

    pdf_links = []
    for each_file in files:
        filename = each_file["title"]
        with open("papers/" + filename, "r") as file:
            try:
                content = file.read()
                # Find all PDF links in the file (both single and double quoted)
                found_links = re.findall(r'''["'](https?://[^"']*\.pdf)["']''', content)
                pdf_links.extend(found_links)
                print(f"{filename} ({each_file["link"]}): found {len(found_links)} PDF link(s)")
                print(model.get_abstract_atoms(client, content))
            except Exception as e:
                print(e)
    return "200"

client = model.genai.Client()

models = model.load_model()

app.run(host='0.0.0.0', port=8080, debug=True)