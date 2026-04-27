from flask import Flask, render_template
from flask_socketio import SocketIO, emit
import llm_providers
import scholar
import model
from dotenv import load_dotenv
import os
load_dotenv()

N_PAPERS = 5
app = Flask(__name__)
app.config["SECRET_KEY"] = os.getenv("SECRET_KEY")
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="gevent")

provider = llm_providers.get_provider("ollama")
print("Provider loaded: ollama")


@app.route("/")
def main():
    return render_template("index.html")


@socketio.on("run_inference")
def inference(data):
    query = data.get("query", "")

    def progress(message, pct):
        emit("progress", {"message": message, "pct": pct})

    progress("Extracting keywords…", 5)
    keywords = model.extract_keywords(provider, query)

    progress(f"Searching Semantic Scholar for: {keywords}", 15)
    papers = scholar.get_papers(keywords)
    progress(f"Found {len(papers)} papers", 30)

    query_atoms = model.split_to_atoms(provider, query)
    progress(f"Split query into {len(query_atoms)} atoms", 40)

    papers_to_use = []
    total = len(papers) * len(query_atoms)
    done = 0

    for paper in papers:
        for query_atom in query_atoms:
            # Compare atom against the abstract (falls back to title if absent)
            target_text = paper.get("abstract") or paper["title"]
            similarity = model.similarity(provider, query_atom, target_text)
            if similarity > 0.6:
                paper["similarity"] = max(paper.get("similarity", 0), similarity)
                if paper not in papers_to_use:
                    papers_to_use.append(paper)

            done += 1
            pct = 40 + int((done / total) * 45)  # 40 → 85 %
            progress(
                f"Scoring '{paper['title'][:40]}…' vs atom '{query_atom}'  ({similarity:.2f})",
                pct,
            )

    papers_to_use.sort(key=lambda d: d["similarity"], reverse=True)
    papers_to_use = papers_to_use[:N_PAPERS]

    progress(f"Ranked top {len(papers_to_use)} papers", 90)

    results = [
        {
            "title": p["title"],
            "abstract": p.get("abstract", ""),
            "url": p.get("url", ""),
            "year": p.get("year"),
            "authors": p.get("authors", []),
            "similarity": round(p["similarity"], 3),
        }
        for p in papers_to_use
    ]

    progress("Done!", 100)
    emit("result", {"papers": results})


if __name__ == "__main__":
    socketio.run(app, host="0.0.0.0", port=8080, debug=True)