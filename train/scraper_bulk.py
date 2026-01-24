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

def get_race_ids(year, month):
    """
    Fetches race IDs for a given year and month.
    URL format: https://race.netkeiba.com/top/calendar.html?year={year}&month={month}
    This is a simplification; handling all race IDs often requires parsing calendar pages.
    """
    # Note: Netkeiba's calendar structure is complex.
    # For this implementation, we will assume a simplified list or limited scope 
    # to demonstrate functionality without overloading the server or complexity.
    # In a full-scale app, we would parse https://db.netkeiba.com/?pid=race_list&year={}&month={}
    
    url = f"https://db.netkeiba.com/?pid=race_list&year={year}&month={month}"
    html = fetch_html(url)
    if not html: return []
    
    soup = BeautifulSoup(html, "lxml")
    race_ids = []
    
    # 1. Find all date links (e.g. /race/list/20230105/ or ?pid=race_list&date=20230105)
    date_links = set()
    for a in soup.find_all("a"):
        href = a.get("href")
        if not href: continue
        
        # New pattern: /race/list/20230105/
        if "/race/list/20" in href:
             # Normalize
             if href.startswith("/"):
                 href = "https://db.netkeiba.com" + href
             date_links.add(href)
             
        # Old pattern: pid=race_list&date=
        elif "pid=race_list&date=" in href:
            if href.startswith("/"):
                href = "https://db.netkeiba.com" + href
            elif not href.startswith("http"):
                href = "https://db.netkeiba.com/?" + href.split("?")[-1]
            date_links.add(href)
            
    print(f"  Found {len(date_links)} race days in {year}-{month}.")
    
    # 2. Visit each date page to get race IDs
    for date_url in sorted(list(date_links)):
        # print(f"  Fetching day: {date_url}")
        d_html = fetch_html(date_url)
        if not d_html: continue
        
        d_soup = BeautifulSoup(d_html, "lxml")
        
        # In the daily list, links are standard /race/ID/
        # Check both /race/ and /race/sum/ if any
        # Also avoid capturing the date list itself if it matches /race/something
        for a in d_soup.select("a[href^='/race/']"):
            href = a.get("href")
            # /race/202306010101/
            parts = href.split("/")
            # filter out /race/list/...
            if "list" in href: continue
            
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

def bulk_scrape(year_start, year_end, month_start=1, month_end=12):
    """
    Main function to scrape a range of data.
    """
    all_data = [] # Keep accumulating if needed for return, or verify memory usage
    
    for year in range(year_start, year_end + 1):
        year_data = []
        for month in range(month_start, month_end + 1):
            print(f"Scraping {year}-{month}...")
            rids = get_race_ids(year, month)
            print(f"Found {len(rids)} races.")
            
            for rid in tqdm(rids):
                data = scrape_race_data(rid)
                if data:
                    year_data.extend(data)
        
        # Save per year
        if year_data:
            df_year = pd.DataFrame(year_data)
            save_path = os.path.join(settings.RAW_DATA_DIR, f"results_{year}.csv")
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
    args = parser.parse_args()
    
    print(f"Starting scrape from {args.start}-{args.month_start} to {args.end}-{args.month_end}...")
    bulk_scrape(args.start, args.end, args.month_start, args.month_end)
