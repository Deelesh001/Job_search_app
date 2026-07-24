import os
import json
import requests
import pandas as pd

# Target endpoints
ARBEITNOW_URL = "https://www.arbeitnow.com/api/job-board-api"
REMOTIVE_URL = "https://remotive.com/api/remote-jobs"

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "raw_jobs_snapshot.json")


def fetch_arbeitnow():
    print("Fetching from Arbeitnow...")
    jobs = []
    url = ARBEITNOW_URL
    
    # Grab first 8 pages to get a decent sample size (8,000 jobs)
    for _ in range(8):
        try:
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            print(f"Arbeitnow request failed: {e}")
            break
            
        for item in data.get("data", []):
            jobs.append({
                "title": item.get("title", ""),
                "company": item.get("company_name", ""),
                "location": item.get("location", ""),
                "description": item.get("description", ""),
                "source": "arbeitnow",
                "url": item.get("url", ""),
                "date_posted": str(item.get("created_at", "")),
                "category": ", ".join(item.get("tags", []))
            })
            
        # Pagination check
        url = data.get("links", {}).get("next")
        if not url:
            break
            
    return jobs


def fetch_remotive():
    print("Fetching from Remotive...")
    jobs = []
    try:
        resp = requests.get(REMOTIVE_URL, timeout=10)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        print(f"Remotive request failed: {e}")
        return jobs

    for item in data.get("jobs", []):
        jobs.append({
            "title": item.get("title", ""),
            "company": item.get("company_name", ""),
            "location": item.get("candidate_required_location", ""),
            "description": item.get("description", ""),
            "source": "remotive",
            "url": item.get("url", ""),
            "date_posted": str(item.get("publication_date", "")),
            "category": item.get("category", "")
        })
        
    return jobs


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    arbeitnow_jobs = fetch_arbeitnow()
    remotive_jobs = fetch_remotive()
    
    all_jobs = arbeitnow_jobs + remotive_jobs
    
    # Drop empty descriptions or missing titles
    df = pd.DataFrame(all_jobs)
    df = df.dropna(subset=["title", "description"])
    df = df[df["description"].str.strip() != ""]
    
    # Deduplicate on URL
    df = df.drop_duplicates(subset=["url"]).reset_index(drop=True)
    
    print(f"Total jobs collected: {len(df)} (Arbeitnow: {len(arbeitnow_jobs)}, Remotive: {len(remotive_jobs)})")
    
    df.to_json(OUTPUT_FILE, orient="records", indent=2)
    print(f"Saved raw snapshot to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()