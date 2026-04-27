import os
import requests
from dotenv import load_dotenv

load_dotenv()

_BASE = "https://api.semanticscholar.org/graph/v1"
_FIELDS = "title,abstract,url,year,authors"

def get_papers(keywords: str, limit: int = 5) -> list[dict]:
    api_key = os.getenv("SEMANTIC_SCHOLAR_API_KEY")
    headers = {"x-api-key": api_key} if api_key else {}

    try:
        resp = requests.get(
            f"{_BASE}/paper/search",
            params={"query": keywords, "limit": limit, "fields": _FIELDS},
            headers=headers,
            timeout=10,
        )
        resp.raise_for_status()
        raw = resp.json().get("data", [])
    except Exception as e:
        print(f"[scholar] Semantic Scholar request failed: {e}")
        return []

    papers = []
    for item in raw:
        papers.append(
            {
                "title":      item.get("title") or "",
                "abstract":   item.get("abstract"),
                "url":        item.get("url") or
                              f"https://www.semanticscholar.org/paper/{item.get('paperId', '')}",
                "year":       item.get("year"),
                "authors":    [a["name"] for a in item.get("authors", [])],
                "similarity": 0.0,
            }
        )
    return papers