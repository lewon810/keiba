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

import argparse
import sys

def get_args():
    parser = argparse.ArgumentParser(description="Scrape or Merge Horse Profiles")
    parser.add_argument("--input", help="Input result CSV file (filename in raw dir or full path) to scrape missing horses from")
    parser.add_argument("--output", help="Output CSV file for scraped profiles")
    parser.add_argument("--target", help="Existing profile database to check against (default: horse_profiles.csv in raw dir)")
    
    parser.add_argument("--merge_source", help="Source CSV to merge into target")
    parser.add_argument("--merge_target", help="Target CSV to update (default: horse_profiles.csv in raw dir)")
    
    return parser.parse_args()

def resolve_path(path_str):
    """Resolves path relative to RAW_DATA_DIR if it's just a filename."""
    if not path_str:
        return None
    if os.path.isabs(path_str) or os.sep in path_str or '/' in path_str:
        return path_str
    return os.path.join(settings.RAW_DATA_DIR, path_str)

def scrape_missing_horses(input_path=None, output_path=None, target_db_path=None):
    """
    Scans raw data for horses, checks against existing profile DB, 
    and scrapes missing ones.
    """
    # Defaults
    if not target_db_path:
        target_db_path = os.path.join(settings.RAW_DATA_DIR, "horse_profiles.csv")
    else:
        target_db_path = resolve_path(target_db_path)
    
    # 1. Gather all unique horse IDs from results
    all_horse_ids = set()
    
    files_to_scan = []
    if input_path:
        # Resolve input path
        full_input_path = resolve_path(input_path)
        if os.path.exists(full_input_path):
            files_to_scan.append(full_input_path)
            print(f"Scanning single file: {full_input_path}")
        else:
            print(f"Error: Input file not found: {full_input_path}")
            return
    else:
        # Scan all
        scan_files = [f for f in os.listdir(settings.RAW_DATA_DIR) if f.startswith('results_') and f.endswith('.csv')]
        print("Scanning all result files for horses...")
        for f in scan_files:
            files_to_scan.append(os.path.join(settings.RAW_DATA_DIR, f))
    
    for path in files_to_scan:
        try:
            df = pd.read_csv(path, usecols=['horse_id'])
            all_horse_ids.update(df['horse_id'].astype(str))
        except Exception as e:
            print(f"Error reading {path}: {e}")
            pass
            
    print(f"Total unique horses found in source: {len(all_horse_ids)}")
    
    # 2. Load existing profiles (Target DB)
    existing_ids = set()
    if os.path.exists(target_db_path):
        try:
            df_prof = pd.read_csv(target_db_path)
            # Ensure columns exist
            if 'horse_id' in df_prof.columns:
                existing_ids = set(df_prof['horse_id'].astype(str))
        except:
            print("Error reading existing profile DB.")
            
    print(f"Existing profiles in target: {len(existing_ids)}")
    
    # 3. Identify missing
    missing_ids = sorted(list(all_horse_ids - existing_ids))
    print(f"Missing profiles to scrape: {len(missing_ids)}")
    
    if not missing_ids:
        print("No new horses to scrape.")
        return

    # Determine output path
    if not output_path:
        if input_path:
             # If input was "results_2020.csv", default output to "scraped_results_2020.csv" or similar?
             # Or just append to target if no output specified?
             # For safety with CLI, let's require output or default to appending to target.
             # Actually current logic appends to 'profile_path' which was target.
             output_path = target_db_path
        else:
             output_path = target_db_path
    
    output_path = resolve_path(output_path)
    print(f"Saving scraped data to: {output_path}")

    # 4. Scrape
    new_data = []
    BUFFER_SIZE = 50
    
    for hid in tqdm(missing_ids):
        data = scrape_horse_profile(hid)
        if data:
            new_data.append(data)
            
        # Incremental Save
        if len(new_data) >= BUFFER_SIZE:
            _append_profiles(new_data, output_path)
            new_data = []
            
    # Final Save
    if new_data:
        _append_profiles(new_data, output_path)

def merge_profiles(source_path, target_path):
    """Merges source CSV into target CSV with deduplication."""
    source_path = resolve_path(source_path)
    target_path = resolve_path(target_path)
    
    if not os.path.exists(source_path):
        print(f"Source file not found: {source_path}")
        return
    
    print(f"Merging {source_path} into {target_path}")
    
    try:
        df_source = pd.read_csv(source_path)
        
        if os.path.exists(target_path):
            df_target = pd.read_csv(target_path)
            # Combine
            df_combined = pd.concat([df_target, df_source])
        else:
            df_combined = df_source
            
        # Deduplicate
        if 'horse_id' in df_combined.columns:
            before = len(df_combined)
            # Keep last (newest) or first? Usually existing data is fine, but maybe we want to keep one. 
            # drop_duplicates keeps 'first' by default.
            # If we trust current DB, we might want to keep that. 
            # But usually we just want *a* record.
            df_combined = df_combined.drop_duplicates(subset=['horse_id'], keep='first')
            after = len(df_combined)
            print(f"Merged. Rows: {before} -> {after} (Dropped {before - after} duplicates)")
        
        df_combined.to_csv(target_path, index=False, encoding='utf-8')
        print("Merge complete.")
        
    except Exception as e:
        print(f"Error during merge: {e}")
        sys.exit(1)

def _append_profiles(data, path):
    if not data: return
    df = pd.DataFrame(data)
    exists = os.path.exists(path)
    # Check if we are appending to a file that might need headers
    # If file exists, mode='a', header=False
    # If file doesn't exist, mode='w' (implicitly), header=True
    
    # Ensuring directory exists
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    df.to_csv(path, mode='a', header=not exists, index=False, encoding='utf-8')

if __name__ == "__main__":
    args = get_args()
    
    if args.merge_source:
        if not args.merge_target:
             args.merge_target = "horse_profiles.csv" # Default relative to raw dir
        merge_profiles(args.merge_source, args.merge_target)
    else:
        scrape_missing_horses(args.input, args.output, args.target)
