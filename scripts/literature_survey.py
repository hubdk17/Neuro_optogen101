"""
literature_survey.py
====================
Systematic Literature Search and Metadata Extraction for Optotagging Methods.
Queries Europe PMC and OpenAlex for relevant optotagging, automated optotagging,
and response reliability papers.
"""

import urllib.request
import urllib.parse
import json
import time
from typing import List, Dict, Any

def search_europe_pmc(query: str, max_results: int = 25) -> List[Dict[str, Any]]:
    url = f"https://www.ebi.ac.uk/europepmc/webservices/rest/search?query={urllib.parse.quote(query)}&format=json&pageSize={max_results}"
    req = urllib.request.Request(url, headers={"User-Agent": "ResearchLiteratureAudit/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data.get("resultList", {}).get("result", [])
    except Exception as e:
        print(f"Error querying Europe PMC for '{query}': {e}")
        return []

def search_openalex(query: str, max_results: int = 25) -> List[Dict[str, Any]]:
    url = f"https://api.openalex.org/works?search={urllib.parse.quote(query)}&per-page={max_results}"
    req = urllib.request.Request(url, headers={"User-Agent": "ResearchLiteratureAudit/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data.get("results", [])
    except Exception as e:
        print(f"Error querying OpenAlex for '{query}': {e}")
        return []

if __name__ == "__main__":
    queries = [
        "optotagging Neuropixels",
        "automated optotagging",
        "SALT Kepecs optotagging",
        "direct indirect optogenetic activation latency",
        "optogenetic response reliability spike"
    ]
    
    print("=== SEARCHING EUROPE PMC ===")
    seen_titles = set()
    all_papers = []
    
    for q in queries:
        print(f"\nQuery: {q}")
        results = search_europe_pmc(q, max_results=10)
        for r in results:
            title = r.get("title", "").strip().rstrip(".")
            if title and title.lower() not in seen_titles:
                seen_titles.add(title.lower())
                author = r.get("authorString", "Unknown")
                year = r.get("pubYear", "Unknown")
                journal = r.get("journalTitle", r.get("source", "Unknown"))
                doi = r.get("doi", "")
                pmid = r.get("pmid", "")
                print(f"[{year}] {author[:30]}... | {title[:60]}... ({journal})")
                all_papers.append({
                    "title": title,
                    "authors": author,
                    "year": year,
                    "venue": journal,
                    "doi": doi,
                    "pmid": pmid,
                    "source": "EuropePMC"
                })
        time.sleep(0.5)
        
    print(f"\nTotal unique papers found: {len(all_papers)}")
