import requests
from bs4 import BeautifulSoup

def fetch_race_data(url):
    """
    Fetches race data from the given netkeiba URL.
    Returns a list of dictionaries containing horse information.
    """
    print(f"Fetching data from: {url}")
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    
    try:
        response = requests.get(url, headers=headers)
        response.encoding = response.apparent_encoding  # Handle Japanese encoding
        
        soup = BeautifulSoup(response.text, "lxml")
        
        # Parse Race Metadata (Shutuba Page)
        metadata = {
            "course_type": "unknown",
            "distance": 0,
            "weather": "cloudy", # Default
            "condition": "good"  # Default
        }
        
        try:
            data01 = soup.select_one("div.RaceData01")
            if data01:
                text = data01.get_text(strip=True) # e.g. "14:20発走 / 芝1800m (右 C)"
                
                # Course / Dist
                if "芝" in text: metadata["course_type"] = "turb"
                elif "ダ" in text: metadata["course_type"] = "dirt"
                elif "障" in text: metadata["course_type"] = "steeple"
                
                import re
                dist_match = re.search(r'\d{4}', text) # e.g. 1800
                if dist_match:
                    metadata["distance"] = int(dist_match.group())
                    
                # Weather & Condition (Often in separate spans or not strictly standard in Shutuba)
                # For MVP, we stick to defaults or try simple parsing if visible text exists
                if "天候:晴" in text or "晴" in text: metadata["weather"] = "sunny"
                elif "雨" in text: metadata["weather"] = "rainy"
                
                if "良" in text: metadata["condition"] = "good"
                elif "重" in text: metadata["condition"] = "heavy"

        except Exception as e:
            print(f"Metadata error: {e}")
        
        rows = soup.select("tr.HorseList")
        if not rows:
            print("No horse rows found. Logic might need adjustment or URL is invalid.")
            return []
            
        race_data = []
        for row in rows:
            horse = metadata.copy()
            
            # Helper to safely extract text
            def get_text(selector):
                element = row.select_one(selector)
                return element.get_text(strip=True) if element else ""
            
            # Helper to extract IDs from href
            def get_id(selector, id_type="horse"):
                element = row.select_one(selector)
                if element and element.get("href"):
                    href = element.get("href")
                    parts = href.split("/")
                    parts = [p for p in parts if p]
                    if parts:
                        return parts[-1]
                return "0"

            horse["umaban"] = get_text("td.Umaban")
            horse["waku"] = get_text("td.Waku")
            horse["name"] = get_text(".HorseName a")
            horse["horse_id"] = get_id(".HorseName a")
            
            horse["jockey"] = get_text("td.Jockey a")
            horse["jockey_id"] = get_id("td.Jockey a")
            
            horse["ninki"] = get_text("td.Popular_Ninki")
            
            # Odds extraction
            odds_span = row.select_one("[id^='odds-']")
            if odds_span:
                horse["odds"] = odds_span.get_text(strip=True)
            else:
                horse["odds"] = get_text("td.Popular")
            
            # If name is empty, it might be a malformed row or different structure, skip or keep
            if horse["name"]:
                race_data.append(horse)
                
        print(f"Parsed {len(race_data)} horses.")
        return race_data

    except Exception as e:
        print(f"Error in fetch_race_data: {e}")
        return []

if __name__ == "__main__":
    # Test run
    data = fetch_race_data("https://race.netkeiba.com/race/shutuba.html?race_id=202606010809&rf=race_list")
    for h in data:
        print(h)
