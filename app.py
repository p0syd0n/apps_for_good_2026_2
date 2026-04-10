#main app file
#import libraries
from flask import Flask, request, render_template
import llm_providers
import scholar
import requests
import re
import model
from dotenv import load_dotenv

# Allow for usage of api keys in .env
load_dotenv()

# Max number of papers used
N_PAPERS = 5

# Create the flask app
app = Flask(__name__)
provider = llm_providers.get_provider("gemini")

# Test route
@app.route('/')
def main():
    print("Main hit")
    return render_template("index.html")

# Inference route
@app.route("/inference")
def inference():
    # Retrieve query
    query = request.args["query"]
    # Extract individual keywords
    keywords = model.extract_keywords(provider, query)
    # Get the papers for these keywords
    papers = scholar.get_papers(keywords)
    # Split the query to individual atoms
    query_atoms = model.split_to_atoms(provider, query)

    # Create the list of papers which are relevant (query atom vs paper title)
    # Sorted by similarity, ascending
    papers_to_use = []
    for paper in papers:
        for query_atom in query_atoms:
            similarity = model.similarity(provider, query_atom, paper["title"])
            if similarity > 0.6:
                paper["similarity"] = similarity
                papers_to_use.append(paper)

    # Sort
    papers_to_use.sort(key=lambda d: d["similarity"], reverse=True)

    # Cut to max number of papers
    if len(papers_to_use) > N_PAPERS:
        papers_to_use = papers_to_use[:N_PAPERS]

    # Fetch HTML and immediately extract PDF links
    pdf_links = []
    for paper in papers_to_use:
        print(paper["title"] + " : " + paper["link"])
        try:
            # Fetch the HTML content directly into memory
            content = requests.get(paper["link"]).content.decode('utf-8')
            # Find all PDF links in the content (both single and double quoted)
            found_links = re.findall(r'''["'](https?://[^"']*\.pdf)["']''', content)
            pdf_links.extend(found_links)
            print(f"{paper['title']} ({paper['link']}): found {len(found_links)} PDF link(s)")
            print(model.get_abstract_atoms(provider, content))
            # Save the PDF links to papers/
            if found_links:
                safe_title = re.sub(r'[^\w\s-]', '', paper['title'])[:100]
                with open(f"papers/{safe_title}.txt", "w") as f:
                    f.write("\n".join(found_links))
        except Exception as e:
            print("Problem with request/parse: " + str(e))

    return "200"

app.run(host='0.0.0.0', port=8080, debug=True)