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
        time.sleep(0.6)
        response = requests.get(url, headers=headers, timeout=10)
        response.encoding = response.apparent_encoding
        return response.text
    except Exception as e:
        print(f"Error fetching {url}: {e}")
        return None

def scrape_horse_profile(horse_id):
    """Scrapes Sire and Broodmare Sire from Netkeiba horse profile."""
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
        table = soup.select_one("table.blood_table")
        if table:
            rows = table.find_all("tr")
            if len(rows) >= 17:
                # Sire: Row 0, Cell 0
                sire_td = rows[0].find("td")
                if sire_td:
                    a = sire_td.find("a")
                    if a: 
                        profile["sire_name"] = a.get_text(strip=True)
                        profile["sire_id"] = a.get("href").split("/")[-2]
                
                # DamSire: Row 16, Cell 1
                dam_row = rows[16]
                cells = dam_row.find_all("td")
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
    parser.add_argument("--input", help="Input result CSV file")
    parser.add_argument("--output", help="Output CSV file")
    parser.add_argument("--target", help="Existing profile database")
    parser.add_argument("--merge_source", help="Source CSV to merge")
    parser.add_argument("--merge_target", help="Target CSV to update")
    return parser.parse_args()

def resolve_path(path_str):
    if not path_str: return None
    if os.path.isabs(path_str) or os.sep in path_str or '/' in path_str:
        return path_str
    return os.path.join(settings.RAW_DATA_DIR, path_str)

def normalize_id(val):
    """正規化: 文字列化 -> 前後空白削除 -> .0削除"""
    s = str(val).strip()
    if s.endswith(".0"):
        return s[:-2]
    return s

def scrape_missing_horses(input_path=None, output_path=None, target_db_path=None):
    if not target_db_path:
        target_db_path = os.path.join(settings.RAW_DATA_DIR, "horse_profiles.csv")
    else:
        target_db_path = resolve_path(target_db_path)
    
    # 1. 結果データからユニークな馬IDを収集
    all_horse_ids = set()
    files_to_scan = []
    
    if input_path:
        full_input_path = resolve_path(input_path)
        if os.path.exists(full_input_path):
            files_to_scan.append(full_input_path)
        else:
            print(f"エラー: 入力ファイルが見つかりません: {full_input_path}")
            return
    else:
        scan_files = [f for f in os.listdir(settings.RAW_DATA_DIR) if f.startswith('results_') and f.endswith('.csv')]
        for f in scan_files:
            files_to_scan.append(os.path.join(settings.RAW_DATA_DIR, f))
    
    for path in files_to_scan:
        try:
            # 【重要】dtype=str を指定
            df = pd.read_csv(path, usecols=['horse_id'], dtype={'horse_id': str})
            ids = df['horse_id'].dropna().apply(normalize_id)
            all_horse_ids.update(ids)
        except Exception as e:
            print(f"ファイル読み込みエラー {path}: {e}")
            pass
            
    print(f"ソース内の全ユニーク馬ID数: {len(all_horse_ids)}")
    
    # 2. 既存プロファイルの読み込み
    existing_ids = set()
    if os.path.exists(target_db_path):
        try:
            # 【重要】dtype=str を指定
            df_prof = pd.read_csv(target_db_path, dtype={'horse_id': str})
            if 'horse_id' in df_prof.columns:
                ids = df_prof['horse_id'].dropna().apply(normalize_id)
                existing_ids = set(ids)
        except Exception as e:
            print(f"既存プロファイルDB読み込みエラー: {e}")
            
    print(f"ターゲット内の既存プロファイル数: {len(existing_ids)}")
    
    # 3. 欠損IDの特定
    missing_ids = sorted(list(all_horse_ids - existing_ids))
    print(f"スクレイピング対象の欠損プロファイル数: {len(missing_ids)}")
    
    if not missing_ids:
        print("新規スクレイピング対象の馬はありません。")
        return

    if not output_path:
        output_path = target_db_path
    output_path = resolve_path(output_path)
    if os.path.exists(output_path):
        os.remove(output_path) # output_pathに既存データがあるなら全て削除する
    
    new_data = []
    BUFFER_SIZE = 50
    
    for hid in tqdm(missing_ids):
        data = scrape_horse_profile(hid)
        if data:
            new_data.append(data)
        if len(new_data) >= BUFFER_SIZE:
            _append_profiles(new_data, output_path)
            new_data = []
            
    if new_data:
        _append_profiles(new_data, output_path)

def merge_profiles(source_path, target_path):
    source_path = resolve_path(source_path)
    target_path = resolve_path(target_path)
    
    if not os.path.exists(source_path):
        return
    
    print(f"マージ中: {source_path} -> {target_path}")
    
    try:
        # 【重要】dtype=str を指定
        df_source = pd.read_csv(source_path, dtype={'horse_id': str})
        if 'horse_id' in df_source.columns:
             df_source['horse_id'] = df_source['horse_id'].dropna().apply(normalize_id)

        if os.path.exists(target_path):
            df_target = pd.read_csv(target_path, dtype={'horse_id': str})
            if 'horse_id' in df_target.columns:
                df_target['horse_id'] = df_target['horse_id'].dropna().apply(normalize_id)
            df_combined = pd.concat([df_target, df_source])
        else:
            df_combined = df_source
            
        if 'horse_id' in df_combined.columns:
            before = len(df_combined)
            df_combined = df_combined.drop_duplicates(subset=['horse_id'], keep='last')
            after = len(df_combined)
            print(f"マージ完了。 行数: {before} -> {after} ({before - after} 件の重複を削除)")
        
        df_combined.to_csv(target_path, index=False, encoding='utf-8')
        print("マージ処理完了。")
        
    except Exception as e:
        print(f"マージエラー: {e}")
        sys.exit(1)

def _append_profiles(data, path):
    if not data: return
    df = pd.DataFrame(data)
    exists = os.path.exists(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, mode='a', header=not exists, index=False, encoding='utf-8')

if __name__ == "__main__":
    args = get_args()
    if args.merge_source:
        if not args.merge_target: args.merge_target = "horse_profiles.csv"
        merge_profiles(args.merge_source, args.merge_target)
    else:
        scrape_missing_horses(args.input, args.output, args.target)
