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
        
        rows = soup.select("tr.HorseList")
        if not rows:
            print("No horse rows found. Logic might need adjustment or URL is invalid.")
            return []
            
        race_data = []
        for row in rows:
            horse = {}
            
            # Helper to safely extract text
            def get_text(selector):
                element = row.select_one(selector)
                return element.get_text(strip=True) if element else ""

            horse["umaban"] = get_text("td.Umaban")
            horse["waku"] = get_text("td.Waku")
            horse["name"] = get_text(".HorseName a")
            horse["jockey"] = get_text("td.Jockey a")
            horse["ninki"] = get_text("td.Popular_Ninki")
            
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
