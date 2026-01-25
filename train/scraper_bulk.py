import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import os
import random
from tqdm import tqdm
from . import settings
import re

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

def get_race_ids(year, month):
    """
    Fetches race IDs for a given year and month.
    Uses race.netkeiba.com/top/calendar.html because it reliably lists specific dates,
    unlike db.netkeiba.com/top/calendar.html which can be messy.
    """
    # 1. Get Calendar Page to find dates
    url = f"https://race.netkeiba.com/top/calendar.html?year={year}&month={month}"
    html = fetch_html(url)
    if not html: return []
    
    soup = BeautifulSoup(html, "lxml")
    date_urls = set()
    target_ym = f"{year}{month:02}"
    
    # 2. Extract race dates (kaisai_date=YYYYMMDD)
    for a in soup.find_all("a"):
        href = a.get("href")
        if not href: continue
        
        # Format usually: ../top/race_list.html?kaisai_date=20230401
        if "kaisai_date" in href:
             match = re.search(r"kaisai_date=(\d{8})", href)
             if match:
                 d_str = match.group(1)
                 # Strict filter to ensure we stay within the requested month
                 # (Calendar view might show adjacent days)
                 if d_str.startswith(target_ym):
                     # Construct db.netkeiba.com list URL
                     # https://db.netkeiba.com/race/list/YYYYMMDD/
                     list_url = f"https://db.netkeiba.com/race/list/{d_str}/"
                     date_urls.add(list_url)
    
    print(f"  Found {len(date_urls)} race days in {year}-{month} (from race.netkeiba.com).")
    
    race_ids = []
    
    # 3. Visit each daily list page on DB to get race IDs
    for date_url in sorted(list(date_urls)):
        d_html = fetch_html(date_url)
        if not d_html: continue
        
        d_soup = BeautifulSoup(d_html, "lxml")
        
        # Link format on list page: /race/202306030301/
        for a in d_soup.select("a[href^='/race/']"):
            href = a.get("href")
            # Filter out non-race links
            if "list" in href: continue
            
            parts = href.split("/")
            # Expect /race/ID/
            if len(parts) >= 3:
                rid = parts[2]
                if rid.isdigit() and len(rid) == 12:
                    race_ids.append(rid)
                    
    return sorted(list(set(race_ids)))

def scrape_race_data(race_id):
    """
    Scrapes result data for a specific race ID from db.netkeiba.com.
    """
    url = f"https://db.netkeiba.com/race/{race_id}/"
    html = fetch_html(url)
    if not html: return None
    
    soup = BeautifulSoup(html, "lxml")
    
    # Parse Race Info (Metadata)
    # usually in div.data_intro > dl.racedata > h1 (Title) and p (Details) or similar structure depending on race/shutuba
    # For db.netkeiba race result page:
    # <dl class="racedata">
    #   <dt>
    #      2016年1月5日 1回中山1日目 3歳未勝利
    #   </dt>
    #   <dd>
    #      芝右1600m / 天候 : 晴 / 芝 : 良 / 発走 : 09:50
    #   </dd>
    # </dl>
    
    race_info = {
        "race_id": race_id,
        "course_type": "",
        "distance": 0,
        "weather": "",
        "condition": "",
        "date": ""
    }
    
    try:
        racedata = soup.select_one("dl.racedata")
        if racedata:
            # Date
            dt_text = racedata.select_one("dt").get_text(strip=True) if racedata.select_one("dt") else ""
            race_info["date"] = dt_text.split(" ")[0] # "2016年1月5日"
            
            # Conditions
            dd_text = racedata.select_one("dd").get_text(strip=True) if racedata.select_one("dd") else ""
            # e.g. "芝右1600m / 天候 : 晴 / 芝 : 良 / 発走 : 09:50"
            
            parts = dd_text.split("/")
            if len(parts) >= 1:
                # Part 0: Course/Dist e.g. "芝右1600m" or "ダ右1800m"
                dist_str = parts[0].strip()
                if "芝" in dist_str: race_info["course_type"] = "turb"
                elif "ダ" in dist_str: race_info["course_type"] = "dirt"
                elif "障" in dist_str: race_info["course_type"] = "steeple"
                
                # Extract number
                import re
                dist_match = re.search(r'\d+', dist_str)
                if dist_match:
                    race_info["distance"] = int(dist_match.group())
            
            if len(parts) >= 2:
                # Weather e.g. "天候 : 晴"
                if "晴" in parts[1]: race_info["weather"] = "sunny"
                elif "曇" in parts[1]: race_info["weather"] = "cloudy"
                elif "雨" in parts[1]: race_info["weather"] = "rainy"
                elif "小雨" in parts[1]: race_info["weather"] = "drizzle"
                elif "雪" in parts[1]: race_info["weather"] = "snow"
                
            if len(parts) >= 3:
                # Condition e.g. "芝 : 良" or "ダート : 重"
                cond_str = parts[2].strip()
                if "良" in cond_str: race_info["condition"] = "good"
                elif "稍重" in cond_str: race_info["condition"] = "slightly_heavy"
                elif "重" in cond_str: race_info["condition"] = "heavy"
                elif "不良" in cond_str: race_info["condition"] = "bad"

    except Exception as e:
        print(f"Error parsing race metadata for {race_id}: {e}")

    # Parse Result Table
    rows = soup.select("table.race_table_01 tr")
    if not rows: return None
    
    results = []
    # Header handling is needed effectively, but assume standard DB format
    # Rnk, Frame, Horse#, ...
    
    for row in rows[1:]: # Skip header
        cols = row.select("td")
        if len(cols) < 10: continue
        
        try:
            res = {
                "race_id": race_id,
                **race_info, # Update with metadata
                "rank": cols[0].get_text(strip=True),
                "waku": cols[1].get_text(strip=True),
                "umaban": cols[2].get_text(strip=True),
                "horse_name": cols[3].get_text(strip=True),
                "horse_id": cols[3].select_one("a").get("href").split("/")[-2] if cols[3].select_one("a") else "",
                "jockey": cols[6].get_text(strip=True),
                "jockey_id": cols[6].select_one("a").get("href").split("/")[-2] if cols[6].select_one("a") else "",
                "time": cols[7].get_text(strip=True),
                "odds": cols[12].get_text(strip=True),
                "popularity": cols[13].get_text(strip=True)
            }
            results.append(res)
        except Exception as e:
            continue
            
    return results

def bulk_scrape(year_start, year_end, month_start=1, month_end=12, force=False):
    """
    Main function to scrape a range of data.
    """
    all_data = [] # Keep accumulating if needed for return, or verify memory usage
    
    for year in range(year_start, year_end + 1):
        year_data = []
        existing_rids = set()
        save_path = os.path.join(settings.RAW_DATA_DIR, f"results_{year}.csv")

        # Check for existing data if not forced
        if not force and os.path.exists(save_path):
            try:
                df_existing = pd.read_csv(save_path)
                # Ensure race_id is string
                if 'race_id' in df_existing.columns:
                    existing_rids = set(df_existing['race_id'].astype(str))
                print(f"File {save_path} exists. Found {len(existing_rids)} existing races.")
                
                # If we are just reading, we might want to keep existing data in memory if we plan to return it
                # But here we focus on appending new data. 
                # Strategy: Load existing data into year_data so we can re-save the complete set later?
                # OR: append mode?
                # Safer: Load all existing into year_data, then append new ones, then save valid complete file.
                year_data = df_existing.to_dict('records')
            except Exception as e:
                print(f"Error reading existing file {save_path}: {e}")

        for month in range(month_start, month_end + 1):
            print(f"Scraping {year}-{month}...")
            rids = get_race_ids(year, month)
            
            # Filter out existing
            new_rids = [rid for rid in rids if rid not in existing_rids]
            print(f"Found {len(rids)} races ({len(new_rids)} new).")
            
            if not new_rids:
                continue

            for rid in tqdm(new_rids):
                data = scrape_race_data(rid)
                if data:
                    year_data.extend(data)
                    # Add to existing_rids to avoid duplicates if month ranges overlap logic somehow? 
                    # (Logic handles distinct rids)
        
        # Save per year
        if year_data:
            df_year = pd.DataFrame(year_data)
            # Ensure safe write
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            df_year.to_csv(save_path, index=False)
            print(f"Saved {len(df_year)} rows to {save_path}")
            all_data.extend(year_data)
            
    return pd.DataFrame(all_data)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Bulk scrape Netkeiba data.")
    parser.add_argument("--start", type=int, default=2023, help="Start year")
    parser.add_argument("--end", type=int, default=2023, help="End year")
    parser.add_argument("--month_start", type=int, default=1, help="Start month")
    parser.add_argument("--month_end", type=int, default=12, help="End month")
    parser.add_argument("--force", action="store_true", help="Force overwrite existing data")
    args = parser.parse_args()
    
    print(f"Starting scrape from {args.start}-{args.month_start} to {args.end}-{args.month_end} (Force: {args.force})...")
    bulk_scrape(args.start, args.end, args.month_start, args.month_end, args.force)
