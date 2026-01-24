from app.scraper import search_races
import requests
from bs4 import BeautifulSoup

def debug_search():
    date_str = "20260124"
    print(f"Testing search for {date_str}...")
    
    # 1. Run properties
    races = search_races(date_str, place_code="06")
    print(f"Found {len(races)} races.")
    
    # 2. Raw HTML Check (Sub Endpoint)
    url = f"https://race.netkeiba.com/top/race_list_sub.html?kaisai_date={date_str}"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "X-Requested-With": "XMLHttpRequest" 
    }
    res = requests.get(url, headers=headers)
    print(f"Status Code: {res.status_code}")
    print(f"Content Length: {len(res.text)}")
    
    soup = BeautifulSoup(res.text, "lxml")
    
    # Check selectors
    links = soup.select("a[href*='race_id=']")
    print(f"Links with race_id=: {len(links)}")
    
    # Check simplified race list
    dl = soup.select("dl.RaceList_DataList")
    print(f"RaceList DL found: {len(dl)}")
    
    # Dump snippet
    with open("debug_search_result.html", "w", encoding="utf-8") as f:
        f.write(res.text)
    print("Saved output to debug_search_result.html")

if __name__ == "__main__":
    debug_search()
