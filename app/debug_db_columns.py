import requests
from bs4 import BeautifulSoup

def debug_columns(race_id):
    url = f"https://db.netkeiba.com/race/{race_id}/"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    print(f"Fetching {url}")
    response = requests.get(url, headers=headers)
    response.encoding = response.apparent_encoding
    
    soup = BeautifulSoup(response.text, "lxml")
    
    # Check headers
    table = soup.select_one("table.race_table_01")
    if not table:
        print("Table not found")
        return
        
    headers = [th.get_text(strip=True) for th in table.select("tr")[0].select("th")]
    print("Headers:")
    for i, h in enumerate(headers):
        print(f"{i}: {h}")
        
    # Check first row
    rows = table.select("tr")
    if len(rows) > 1:
        cols = rows[1].select("td")
        print("\nFirst Row Data:")
        for i, c in enumerate(cols):
            print(f"{i}: {c.get_text(strip=True)}")

if __name__ == "__main__":
    # Use a likely valid recent race ID (e.g., Arima Kinen 2024 or similar, or just a random one)
    # 202406050911 (Arima Kinen 2024? No, let's use a generic one from 2023)
    # 202306010101 (2023 Nakayama Day 1 Race 1)
    debug_columns("202306010101")
