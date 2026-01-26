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

            # Use partial match for Waku/Umaban as classes can be 'Waku1', 'Umaban3' etc.
            horse["umaban"] = get_text("td[class*='Umaban']")
            horse["waku"] = get_text("td[class*='Waku']")
            horse["name"] = get_text(".HorseName a")
            horse["horse_id"] = get_id(".HorseName a")
            
            horse["jockey"] = get_text("td.Jockey a")
            horse["jockey_id"] = get_id("td.Jockey a")
            
            horse["trainer"] = get_text("td.Trainer a")
            horse["trainer_id"] = get_id("td.Trainer a")
            
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
        
        # --- Fetch Real-time Odds via API ---
        # Extract race_id from URL
        import re
        rid_match = re.search(r'race_id=(\d+)', url)
        if rid_match:
            rid = rid_match.group(1)
            odds_map = fetch_odds(rid)
            if odds_map:
                print(f"Merged {len(odds_map)} odds records.")
                for horse in race_data:
                    u = horse.get("umaban")
                    if u:
                        u = u.strip()
                        # API uses zero-padded strings (e.g. "01")
                        if len(u) == 1 and u.isdigit():
                            u = u.zfill(2)
                    
                    if u and u in odds_map:
                        horse["odds"] = odds_map[u]
        
        return race_data

    except Exception as e:
        print(f"Error in fetch_race_data: {e}")
        return []

def fetch_odds(race_id):
    """
    Fetch real-time odds from Netkeiba API.
    Returns dict: {umaban: win_odds}
    """
    # Type 1 = Tanfuku (Win/Place)
    api_url = f"https://race.netkeiba.com/api/api_get_jra_odds.html?race_id={race_id}&type=1&action=init"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko)",
        "X-Requested-With": "XMLHttpRequest",
        "Referer": f"https://race.netkeiba.com/odds/index.html?race_id={race_id}"
    }
    
    try:
        res = requests.get(api_url, headers=headers)
        data = res.json()
        
        # 'middle' status also contains valid odds (interim)
        valid_statuses = ['true', 'middle']
        if data.get('status') in valid_statuses and 'data' in data:
            # Structure: data['data']['odds']['1'][umaban] = [Win, Place, Pop...]
            # '1' key under 'odds' likely represents the type (Tanfuku)
            
            odds_data = data['data'].get('odds', {}).get('1', {})
            result = {}
            for umaban, values in odds_data.items():
                # values[0] is Win Odds
                win_odds = values[0]
                result[umaban] = win_odds
            return result
            
    except Exception as e:
        print(f"Odds API Error: {e}")
        
    return {}

if __name__ == "__main__":
    # Test run
    # data = fetch_race_data("https://race.netkeiba.com/race/shutuba.html?race_id=202606010809&rf=race_list")
    # for h in data:
    #     print(h)
    pass

def search_races(date_str, place_code=None, race_no=None):
    """
    Search for race IDs/URLs on a specific date.
    date_str: YYYYMMDD (e.g. "20260124")
    place_code: Optional filter (e.g. "06" for Nakayama)
    race_no: Optional filter (e.g. 11)
    
    Returns: List of dicts {'id': race_id, 'url': url, 'title': title}
    """
    url = f"https://race.netkeiba.com/top/race_list_sub.html?kaisai_date={date_str}"
    print(f"Searching races at: {url}")
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    
    try:
        response = requests.get(url, headers=headers)
        # race_list_sub is UTF-8, unlike main race pages
        response.encoding = 'utf-8' 
             
        soup = BeautifulSoup(response.text, "lxml")
        
        found_races = []
        
        # Selector modification:
        # Sometimes links are just "../race/..." or "/race/..."
        # And sometimes soup selector "a[href*='race_id=']" is too strict if params are different.
        # Let's try broader "a" and filter in loop.
        links = soup.find_all("a")
        seen_ids = set()
        
        for a in links:
            href = a.get("href")
            if not href: continue
            
            # Match race ID pattern in URL: /race/202606010801/ or ?race_id=2026...
            # Shutuba Page search
            
            rid = None
            if "race_id=" in href:
                 # Support both shutuba.html and result.html
                 # href format: ../race/result.html?race_id=202306010201&rf=race_list
                 try:
                     rid = href.split("race_id=")[1].split("&")[0]
                 except: pass
            elif "/race/" in href:
                 # /race/202601010101/ format
                 parts = href.split("/")
                 for p in parts:
                     if p.isdigit() and len(p) == 12:
                         rid = p
                         break
            
            if rid:
                try:
                    
                    if rid in seen_ids: continue
                    
                    # Filtering
                    # rid format YYYYPP...
                    p_code = rid[4:6]
                    r_no = int(rid[-2:])
                    
                    if place_code and p_code != str(place_code).zfill(2):
                        continue
                    if race_no and r_no != int(race_no):
                        continue
                        
                    # Construct full URL
                    # Usually "https://race.netkeiba.com/race/shutuba.html?race_id={rid}"
                    full_url = f"https://race.netkeiba.com/race/shutuba.html?race_id={rid}&rf=race_list"
                    
                    # Get title/metadata from text if available
                    title = a.get_text(strip=True)
                    # title often includes race number and name e.g. "1R 3歳未勝利"
                    
                    # Append
                    found_races.append({
                        "id": rid,
                        "url": full_url,
                        "title": f"{p_code} {r_no}R {title}",
                        "race_no": r_no
                    })
                    seen_ids.add(rid)
                    
                except:
                    continue
                    
        # Sort by race_no (approx)
        found_races.sort(key=lambda x: x['id'])
        return found_races

    except Exception as e:
        print(f"Search error: {e}")
        return []
