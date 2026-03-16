import os
from serpapi import GoogleSearch


SERPAPI_KEY = os.getenv("SERPAPI_KEY")


def search_web(query):

    try:

        params = {
            "engine": "google",
            "q": query,
            "api_key": SERPAPI_KEY,
            "num": 5
        }

        search = GoogleSearch(params)
        results = search.get_dict()

        snippets = []
        sources = []

        for r in results.get("organic_results", [])[:5]:

            snippet = r.get("snippet", "")
            link = r.get("link", "")

            if snippet:
                snippets.append(snippet)

            if link:
                sources.append(link)

        context = "\n".join(snippets)

        return context, sources

    except Exception:

        return "", []