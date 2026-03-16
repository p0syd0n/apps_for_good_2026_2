import serpapi
from dotenv import load_dotenv
import os
load_dotenv()

def get_papers(query):

    client = serpapi.Client(api_key=os.getenv("API_KEY")) 

    results = client.search(engine='google', q=query)
    results = results["organic_results"]

    returned = []
    for result in results:
        print(result)
        returned.append({"title": result["title"], "link": result["link"]})
    return returned


