from bs4 import BeautifulSoup

def verify_parsing():
    try:
        with open("debug.html", "r", encoding="utf-8") as f:
            html = f.read()
            
        soup = BeautifulSoup(html, "lxml")
        
        # Try to find the ShutubaTable
        rows = soup.select("tr.HorseList")
        print(f"Found {len(rows)} horse rows.")
        
        horses = []
        for row in rows:
            horse_data = {}
            
            # Waku (Frame Number)
            waku_td = row.select_one("td.Waku")
            if waku_td:
                horse_data["waku"] = waku_td.get_text(strip=True)
                
            # Umaban (Horse Number)
            umaban_td = row.select_one("td.Umaban")
            if umaban_td:
                horse_data["umaban"] = umaban_td.get_text(strip=True)
                
            # Horse Name
            name_tag = row.select_one("div.HorseName a")
            if name_tag:
                horse_data["name"] = name_tag.get_text(strip=True)
                
            # Jockey
            jockey_td = row.select_one("td.Jockey a")
            if jockey_td:
                horse_data["jockey"] = jockey_td.get_text(strip=True)
                
            horses.append(horse_data)
            
        for i, h in enumerate(horses):
            print(f"{i+1}: {h}")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    verify_parsing()
