import pandas as pd
import requests
from bs4 import BeautifulSoup
import time
import random
import os
from tqdm import tqdm

DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'learn', 'data', 'raw', 'results_2016_2025.csv')
UPDATED_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'learn', 'data', 'raw', 'results_validation_patched.csv')

def fetch_race_odds_pop(race_id):
    url = f"https://db.netkeiba.com/race/{race_id}/"
    headers = {
         "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    try:
        time.sleep(1)
        response = requests.get(url, headers=headers)
        response.encoding = response.apparent_encoding
        soup = BeautifulSoup(response.text, "lxml")
        
        rows = soup.select("table.race_table_01 tr")
        if not rows: return {}
        
        # Map horse_id to (odds, popularity)
        data_map = {}
        for row in rows[1:]:
            cols = row.select("td")
            if len(cols) > 13:
                # horse_id index could be tricky if we don't parse href, but we can try to match by horse name or umaban
                # Let's match by Umaban (Column 2, index 2)
                umaban = cols[2].get_text(strip=True)
                odds = cols[12].get_text(strip=True)
                pop = cols[13].get_text(strip=True)
                data_map[umaban] = (odds, pop)
        return data_map
    except Exception as e:
        print(f"Error {race_id}: {e}")
        return {}

def patch_validation():
    print("Loading original data...")
    df = pd.read_csv(DATA_PATH)
    
    # Identify validation set (Last 20% of Race IDs)
    unique_race_ids = df['race_id'].unique()
    train_size = int(len(unique_race_ids) * 0.8)
    val_race_ids = unique_race_ids[train_size:]
    
    print(f"Validation Set: {len(val_race_ids)} races.")
    
    # We will iterate over val_race_ids and fetch data
    # Results will be updated in a copy of df
    df_patched = df.copy()
    
    # Ensure columns exist and are object type to hold strings
    if 'odds' not in df_patched.columns: df_patched['odds'] = None
    if 'popularity' not in df_patched.columns: df_patched['popularity'] = None
    df_patched['odds'] = df_patched['odds'].astype(object)
    df_patched['popularity'] = df_patched['popularity'].astype(object)

    for rid in tqdm(val_race_ids):
        # Fetch
        data_map = fetch_race_odds_pop(rid)
        
        # Update rows
        # Filter rows for this race
        mask = df_patched['race_id'] == rid
        
        # Iterate over rows in this race to match umaban
        # This is slow row-by-row but safe. Vectorization is hard with dictionary map.
        for idx, row in df_patched[mask].iterrows():
            umaban = str(row['umaban'])
            if umaban in data_map:
                odds, pop = data_map[umaban]
                df_patched.at[idx, 'odds'] = odds
                df_patched.at[idx, 'popularity'] = pop
                
    # Save patched
    df_patched.to_csv(UPDATED_PATH, index=False)
    print(f"Saved patched data to {UPDATED_PATH}")

if __name__ == "__main__":
    patch_validation()
