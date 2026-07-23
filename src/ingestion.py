import os
import json
import time
import requests
import pandas as pd

ARBEITNOW_URL = "https://www.arbeitnow.com/api/job-board-api"
REMOTIVE_URL = "https://remotive.com/api/remote-jobs"

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "raw_jobs_snapshot.json")


def fetch_arbeitnow(max_pages=10):
    print("Fetching from Arbeitnow...")
    jobs = []
    url = ARBEITNOW_URL
    
    # Custom User-Agent prevents basic bot blocking
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) StudentIRProject/1.0"}
    
    for page in range(1, max_pages + 1):
        try:
            print(f"  -> Grabbing page {page}...")
            resp = requests.get(url, headers=headers, timeout=10)
            
            # If rate limited, wait 5 seconds and retry once
            if resp.status_code == 429:
                print("  [!] Rate limit hit (429). Sleeping 5 seconds...")
                time.sleep(5)
                resp = requests.get(url, headers=headers, timeout=10)
                
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            print(f"Arbeitnow fetch stopped at page {page}: {e}")
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
            
        url = data.get("links", {}).get("next")
        if not url:
            break
            
        # 1.5 second pause between requests to prevent 429 errors
        time.sleep(1.5)
            
    return jobs


def fetch_remotive():
    print("Fetching from Remotive...")
    jobs = []
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) StudentIRProject/1.0"}
    
    try:
        resp = requests.get(REMOTIVE_URL, headers=headers, timeout=10)
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
    
    arbeitnow_jobs = fetch_arbeitnow(max_pages=10)
    remotive_jobs = fetch_remotive()
    
    all_jobs = arbeitnow_jobs + remotive_jobs
    
    df = pd.DataFrame(all_jobs)
    df = df.dropna(subset=["title", "description"])
    df = df[df["description"].str.strip() != ""]
    
    df = df.drop_duplicates(subset=["url"]).reset_index(drop=True)
    
    print(f"\nTotal jobs collected: {len(df)} (Arbeitnow: {len(arbeitnow_jobs)}, Remotive: {len(remotive_jobs)})")
    
    df.to_json(OUTPUT_FILE, orient="records", indent=2)
    print(f"Saved raw snapshot to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()