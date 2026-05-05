"""
Fetch book metadata from Open Library for Indian-author / Indian-literature works.

Strategy:
  1. Pull /subjects/{subj}.json for a curated list of Indian-related subjects.
  2. Supplement with /search.json?author={name} for well-known Indian authors.
  3. Optionally enrich each work with /works/{key}.json for the description
     (Open Library doesn't include descriptions in subject/search responses).

Run:
    python data_fetcher.py
Outputs: data/books.csv
"""
from __future__ import annotations
import csv
import json
import time
from pathlib import Path
from typing import Dict, List, Set

import requests

BASE_URL = "https://openlibrary.org"

# Subjects vary in how Open Library indexes them; we try several spellings.
# The /subjects/{slug}.json endpoint returns a "works" array of up to 'limit' items.
INDIAN_SUBJECTS = [
    "indic_literature",
    "indian_literature",
    "indian_fiction",
    "hindi_literature",
    "bengali_literature",
    "tamil_literature",
    "malayalam_literature",
    "marathi_literature",
    "urdu_literature",
    "indian_poetry",
    "indian_authors",
    "literature__india",
    "fiction__india",
    "indic_fiction",
]

# Curated list of Indian authors used as a complementary signal.
# Feel free to extend — wider net = richer corpus, more Open Library calls.
INDIAN_AUTHORS = [
    "R. K. Narayan", "Salman Rushdie", "Arundhati Roy", "Vikram Seth",
    "Amitav Ghosh", "Kiran Desai", "Jhumpa Lahiri", "Rabindranath Tagore",
    "Ruskin Bond", "Anita Desai", "Khushwant Singh", "Mulk Raj Anand",
    "Raja Rao", "U. R. Ananthamurthy", "Mahasweta Devi", "Premchand",
    "Sarat Chandra Chattopadhyay", "Bankim Chandra Chatterjee",
    "Bibhutibhushan Bandyopadhyay", "Aravind Adiga", "Chetan Bhagat",
    "Jeet Thayil", "Manu Joseph", "Rohinton Mistry", "Shashi Tharoor",
    "Vikas Swarup", "Kamala Markandaya", "Kamala Das", "Nayantara Sahgal",
    "Shobhaa De", "Sudha Murty", "Devdutt Pattanaik", "Amish Tripathi",
    "Ashwin Sanghi", "Anuja Chauhan", "Preeti Shenoy", "Twinkle Khanna",
    "Vikram Chandra", "Pankaj Mishra", "Upamanyu Chatterjee",
    "Githa Hariharan", "Easterine Kire", "Perumal Murugan", "Benyamin",
    "M. T. Vasudevan Nair", "O. V. Vijayan", "Mahasweta Devi",
    "Kalki Krishnamurthy", "Sujatha", "Ashokamitran",
]

HEADERS = {"User-Agent": "smai-book-recommender/0.1 (educational project)"}


def fetch_subject(subject: str, limit: int = 200) -> List[Dict]:
    url = f"{BASE_URL}/subjects/{subject}.json"
    try:
        r = requests.get(url, params={"limit": limit}, headers=HEADERS, timeout=30)
        if r.status_code == 404:
            return []
        r.raise_for_status()
        return r.json().get("works", []) or []
    except Exception as e:
        print(f"  [warn] subject={subject} -> {e}")
        return []


def search_author(author: str, limit: int = 30) -> List[Dict]:
    url = f"{BASE_URL}/search.json"
    try:
        r = requests.get(
            url, params={"author": author, "limit": limit}, headers=HEADERS, timeout=30
        )
        r.raise_for_status()
        return r.json().get("docs", []) or []
    except Exception as e:
        print(f"  [warn] author={author} -> {e}")
        return []


def fetch_work_details(work_key: str) -> Dict:
    """Get description + subjects from /works/{key}.json."""
    if not work_key.startswith("/works/"):
        work_key = f"/works/{work_key.lstrip('/')}"
    try:
        r = requests.get(f"{BASE_URL}{work_key}.json", headers=HEADERS, timeout=30)
        r.raise_for_status()
        data = r.json()
        desc = data.get("description", "")
        if isinstance(desc, dict):
            desc = desc.get("value", "")
        return {
            "description": (desc or "")[:2000],
            "subjects": data.get("subjects", []) or [],
        }
    except Exception:
        return {"description": "", "subjects": []}


def normalize_subject_record(rec: Dict, source: str) -> Dict:
    authors = rec.get("authors", []) or []
    cover_id = rec.get("cover_id")
    return {
        "work_key": rec.get("key", ""),
        "title": (rec.get("title") or "").strip(),
        "authors": ", ".join(a.get("name", "") for a in authors).strip(", "),
        "first_publish_year": rec.get("first_publish_year") or "",
        "subjects": "; ".join(rec.get("subject", []) or [])[:1000],
        "description": "",
        "cover_url": f"https://covers.openlibrary.org/b/id/{cover_id}-M.jpg" if cover_id else "",
        "source": f"subject:{source}",
    }


def normalize_search_record(rec: Dict, source: str) -> Dict:
    cover_id = rec.get("cover_i")
    return {
        "work_key": rec.get("key", ""),
        "title": (rec.get("title") or "").strip(),
        "authors": ", ".join(rec.get("author_name", []) or []),
        "first_publish_year": rec.get("first_publish_year") or "",
        "subjects": "; ".join(rec.get("subject", []) or [])[:1000],
        "description": "",
        "cover_url": f"https://covers.openlibrary.org/b/id/{cover_id}-M.jpg" if cover_id else "",
        "source": f"author:{source}",
    }


def main(
    out_path: str = "data/books.csv",
    enrich: bool = True,
    max_books: int = 1500,
    sleep: float = 0.25,
):
    Path("data").mkdir(exist_ok=True)
    seen: Set[str] = set()
    records: List[Dict] = []

    print("=== Pass 1: subjects ===")
    for subj in INDIAN_SUBJECTS:
        works = fetch_subject(subj, limit=200)
        added = 0
        for w in works:
            key = w.get("key", "")
            if key and key not in seen:
                seen.add(key)
                records.append(normalize_subject_record(w, subj))
                added += 1
        print(f"  {subj}: +{added} (total {len(records)})")
        time.sleep(sleep)

    print(f"\n=== Pass 2: authors === (cap {max_books})")
    for a in INDIAN_AUTHORS:
        if len(records) >= max_books:
            break
        docs = search_author(a, limit=30)
        added = 0
        for d in docs:
            key = d.get("key", "")
            if key and key not in seen:
                seen.add(key)
                records.append(normalize_search_record(d, a))
                added += 1
        print(f"  {a}: +{added} (total {len(records)})")
        time.sleep(sleep)

    records = [r for r in records if r["title"]][:max_books]
    print(f"\nUnique works after dedup: {len(records)}")

    if enrich:
        print("\n=== Pass 3: descriptions (slow — one /works/ call per book) ===")
        for i, rec in enumerate(records):
            if i % 50 == 0:
                print(f"  enriching {i}/{len(records)}")
            details = fetch_work_details(rec["work_key"])
            rec["description"] = details["description"]
            if details["subjects"] and not rec["subjects"]:
                rec["subjects"] = "; ".join(details["subjects"])[:1000]
            time.sleep(0.05)

    fields = ["work_key", "title", "authors", "first_publish_year",
              "subjects", "description", "cover_url", "source"]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(records)
    print(f"\nWrote {len(records)} rows -> {out_path}")


if __name__ == "__main__":
    main()