import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import os
import random
from tqdm import tqdm
from . import settings

def fetch_html(url):
    """Fetches HTML with simple retry and random sleep."""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    try:
        time.sleep(1 + random.random()) # Be polite
        response = requests.get(url, headers=headers, timeout=10)
        response.encoding = response.apparent_encoding
        return response.text
    except Exception as e:
        print(f"Error fetching {url}: {e}")
        return None

def scrape_horse_profile(horse_id):
    """
    Scrapes Sire and Broodmare Sire from Netkeiba horse profile.
    URL: https://db.netkeiba.com/horse/ped/{horse_id}/
    """
    url = f"https://db.netkeiba.com/horse/ped/{horse_id}/"
    html = fetch_html(url)
    if not html: return None
    
    soup = BeautifulSoup(html, "lxml")
    
    profile = {
        "horse_id": horse_id,
        "sire_id": "unknown",
        "sire_name": "unknown",
        "damsire_id": "unknown",
        "damsire_name": "unknown"
    }
    
    try:
        # Blood table is usually div.blood_table or class="blood_table"
        # Netkeiba structure:
        # <table> class="blood_table"
        # Row 0, Col 0: Sire (Father)
        # Row 1, Col 0: Dam (Mother)
        # We want Sire (Full) and Dam's Sire (Broodmare Sire)
        
        # Sire: 0-0
        # DamSire: 1-0 -> Father
        
        table = soup.select_one("table.blood_table")
        if table:
            rows = table.find_all("tr")
            
            # Sire is in the first row, usually spanning multiple rows
            # simple lookup: find first 'td' with link
            
            # Let's try selecting by link structure if possible, but structure varies.
            # Standard 5-generation table.
            
            # Sire (Father) - Top-left cell
            sire_node = rows[0].select_one("td a")
            if sire_node:
                profile["sire_name"] = sire_node.get_text(strip=True)
                href = sire_node.get("href") # /horse/00000000/
                if href:
                    profile["sire_id"] = href.split("/")[-2]
            
            # Dam is row 16 (in 5-gen table, it's the bottom half)
            # Actually, let's look for Broodmare Sire (Mother's Father).
            # Structure: 
            # 0: Sire
            # 16: Dam
            #  -> Inside Dam's section, top is DamSire.
            
            # Easier approach: Get all links in the table and map positions? No.
            # Row index strategy for 5-generation table:
            # 0: Sire
            # 16: Dam
            # 16: Dam -> row[16] col 0 is Dam.
            # DamSire is Dam's Father.
            # In the table structure, DamSire is usually at row 16, col 1 (if col 0 is Dam).
            # Wait, `rowspan` makes this tricky.
            
            # Alternative: text based search is unreliable.
            
            # Layout:
            # [Sire] ...
            # [Dam ] [DamSire] ...
            
            # Usually:
            # tr[0] td[0] = Sire
            # tr[16] td[0] = Dam (spanning) -> td[1] ? No.
            
            # Let's use a simpler heuristic.
            # Inspect URLs.
            all_links = table.select("a[href^='/horse/']")
            # 0: Sire
            # 1: Sire's Sire
            # ...
            # It's hard to predict index without parsing table structure.
            
            # Let's rely on `rowspan` parsing if we want to be precise, OR use `db.netkeiba.com/horse/{id}` profile page instead of `ped`.
            # Profile page usually lists: "父:", "母:"
            # But we want DamSire. "母父" is often listed in "血統情報" or similar text.
            pass

    except Exception as e:
        print(f"Error parsing table for {horse_id}: {e}")
        
    # Retry with Main Profile Page which is simpler for direct parents
    # https://db.netkeiba.com/horse/{horse_id}/
    # dl.racedata or table.db_prof_table
    # Actually, main page might not listing IDs nicely.
    # Back to `ped` page.
    
    # Robust scrape for pedigee table:
    # Top Left cell = Sire.
    # Bottom Left cell (visually) = Dam.
    # The cell to the right of Dam is DamSire.
    
    # Let's try to grab all 62(ish) horses and deduce.
    # OR: Just grab "Sire" and "Dam".
    # DamSire is important? Yes ("母父").
    
    # Netkeiba:
    # tr[0] td[0] [rowspan=16] -> Sire
    # tr[16] td[0] [rowspan=16] -> Dam
    # Inside Dam's block:
    #   tr[16] td[1] [rowspan=8] -> DamSire
    
    try:
        table = soup.select_one("table.blood_table")
        if table:
            rows = table.find_all("tr")
            if len(rows) >= 17: # Ensure table size
                # Sire: Row 0, Cell 0
                sire_td = rows[0].find("td")
                if sire_td:
                    a = sire_td.find("a")
                    if a: 
                        profile["sire_name"] = a.get_text(strip=True)
                        profile["sire_id"] = a.get("href").split("/")[-2]
                
                # DamSire: Row 16, Cell 1 (since Cell 0 is Dam usually? No, Dam is Cell 0)
                # Wait, if Dam is at Row 16, Cell 0.
                # Then DamSire is Row 16, Cell 1.
                
                dam_row = rows[16]
                cells = dam_row.find_all("td")
                
                # Cell 0 is Dam
                # Cell 1 is DamSire
                if len(cells) > 1:
                    ds_td = cells[1]
                    a = ds_td.find("a")
                    if a:
                        profile["damsire_name"] = a.get_text(strip=True)
                        profile["damsire_id"] = a.get("href").split("/")[-2]
                        
    except Exception as e:
        pass

    return profile

def scrape_missing_horses():
    """
    Scans raw data for horses, checks against existing profile DB, 
    and scrapes missing ones.
    """
    # 1. Gather all unique horse IDs from results
    all_horse_ids = set()
    files = [f for f in os.listdir(settings.RAW_DATA_DIR) if f.startswith('results_') and f.endswith('.csv')]
    
    print("Scanning result files for horses...")
    for f in files:
        path = os.path.join(settings.RAW_DATA_DIR, f)
        try:
            df = pd.read_csv(path, usecols=['horse_id'])
            all_horse_ids.update(df['horse_id'].astype(str))
        except:
            pass
            
    print(f"Total unique horses found: {len(all_horse_ids)}")
    
    # 2. Load existing profiles
    profile_path = os.path.join(settings.RAW_DATA_DIR, "horse_profiles.csv")
    existing_ids = set()
    if os.path.exists(profile_path):
        try:
            df_prof = pd.read_csv(profile_path)
            # Ensure columns exist
            if 'horse_id' in df_prof.columns:
                existing_ids = set(df_prof['horse_id'].astype(str))
        except:
            print("Error reading existing profile DB.")
            
    print(f"Existing profiles: {len(existing_ids)}")
    
    # 3. Identify missing
    missing_ids = sorted(list(all_horse_ids - existing_ids))
    print(f"Missing profiles: {len(missing_ids)}")
    
    if not missing_ids:
        print("No new horses to scrape.")
        return

    # 4. Scrape
    new_data = []
    BUFFER_SIZE = 50
    
    for hid in tqdm(missing_ids):
        data = scrape_horse_profile(hid)
        if data:
            new_data.append(data)
            
        # Incremental Save
        if len(new_data) >= BUFFER_SIZE:
            _append_profiles(new_data, profile_path)
            new_data = []
            
    # Final Save
    if new_data:
        _append_profiles(new_data, profile_path)

def _append_profiles(data, path):
    if not data: return
    df = pd.DataFrame(data)
    exists = os.path.exists(path)
    df.to_csv(path, mode='a', header=not exists, index=False, encoding='utf-8')

if __name__ == "__main__":
    scrape_missing_horses()
