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

def normalize_id(val):
    s = str(val)
    if s.endswith(".0"):
        return s[:-2]
    return s

def scrape_missing_horses(input_path=None, output_path=None, target_db_path=None):
    """
    生データから馬IDをスキャンし、既存のプロファイルDBと照合して
    欠損している馬のプロファイルをスクレイピングします。
    """
    # デフォルト設定
    if not target_db_path:
        target_db_path = os.path.join(settings.RAW_DATA_DIR, "horse_profiles.csv")
    else:
        target_db_path = resolve_path(target_db_path)
    
    # 1. 結果データからユニークな馬IDを収集
    all_horse_ids = set()
    
    files_to_scan = []
    if input_path:
        # 入力パスの解決
        full_input_path = resolve_path(input_path)
        if os.path.exists(full_input_path):
            files_to_scan.append(full_input_path)
            print(f"単一ファイルをスキャン中: {full_input_path}")
        else:
            print(f"エラー: 入力ファイルが見つかりません: {full_input_path}")
            return
    else:
        # 全ファイルをスキャン
        scan_files = [f for f in os.listdir(settings.RAW_DATA_DIR) if f.startswith('results_') and f.endswith('.csv')]
        print("全結果ファイルから馬IDをスキャン中...")
        for f in scan_files:
            files_to_scan.append(os.path.join(settings.RAW_DATA_DIR, f))
    
    for path in files_to_scan:
        try:
            # horse_idを文字列として読み込むか、floatの可能性を考慮して変換
            df = pd.read_csv(path, usecols=['horse_id'])
            # floatとして読み込まれた場合の対策 (例: 2016.0 -> 2016 -> '2016')に加え、文字列IDも考慮
            ids = df['horse_id'].dropna().apply(normalize_id)
            all_horse_ids.update(ids)
        except Exception as e:
            print(f"ファイル読み込みエラー {path}: {e}")
            pass
            
    print(f"ソース内の全ユニーク馬ID数: {len(all_horse_ids)}")
    
    # 2. 既存プロファイルの読み込み (ターゲットDB)
    existing_ids = set()
    if os.path.exists(target_db_path):
        try:
            df_prof = pd.read_csv(target_db_path)
            # カラム存在確認
            if 'horse_id' in df_prof.columns:
                # 既存DBも念のため同様に正規化
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

    # 出力パスの決定
    if not output_path:
        # 出力が指定されていない場合はターゲットDBに追加する形をとる
        output_path = target_db_path
    
    output_path = resolve_path(output_path)
    print(f"スクレイピングデータを保存: {output_path}")

    # 4. スクレイピング実行
    new_data = []
    BUFFER_SIZE = 50
    
    for hid in tqdm(missing_ids):
        data = scrape_horse_profile(hid)
        if data:
            new_data.append(data)
            
        # 逐次保存
        if len(new_data) >= BUFFER_SIZE:
            _append_profiles(new_data, output_path)
            new_data = []
            
    # 最終保存
    if new_data:
        _append_profiles(new_data, output_path)

def merge_profiles(source_path, target_path):
    """ソースCSVをターゲットCSVにマージし、重複を排除します。"""
    source_path = resolve_path(source_path)
    target_path = resolve_path(target_path)
    
    if not os.path.exists(source_path):
        print(f"ソースファイルが見つかりません: {source_path}")
        return
    
    print(f"マージ中: {source_path} -> {target_path}")
    
    try:
        # IDを明示的に文字列として扱うのではなく、正規化してから処理する
        df_source = pd.read_csv(source_path)
        # ID正規化
        if 'horse_id' in df_source.columns:
             df_source['horse_id'] = df_source['horse_id'].dropna().apply(normalize_id)

        if os.path.exists(target_path):
            df_target = pd.read_csv(target_path)
            if 'horse_id' in df_target.columns:
                df_target['horse_id'] = df_target['horse_id'].dropna().apply(normalize_id)
            
            # 結合
            df_combined = pd.concat([df_target, df_source])
        else:
            df_combined = df_source
            
        # 重複排除
        if 'horse_id' in df_combined.columns:
            before = len(df_combined)
            # 後勝ち (ソース側のデータを優先)
            df_combined = df_combined.drop_duplicates(subset=['horse_id'], keep='last')
            after = len(df_combined)
            print(f"マージ完了。 行数: {before} -> {after} ({before - after} 件の重複を削除)")
        
        df_combined.to_csv(target_path, index=False, encoding='utf-8')
        print("マージ処理完了。")
        
    except Exception as e:
        print(f"マージ中にエラーが発生しました: {e}")
        sys.exit(1)

def _append_profiles(data, path):
    if not data: return
    df = pd.DataFrame(data)
    exists = os.path.exists(path)
    
    # ディレクトリ作成確認
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    df.to_csv(path, mode='a', header=not exists, index=False, encoding='utf-8')

if __name__ == "__main__":
    args = get_args()
    
    if args.merge_source:
        if not args.merge_target:
             args.merge_target = "horse_profiles.csv" # デフォルト
        merge_profiles(args.merge_source, args.merge_target)
    else:
        scrape_missing_horses(args.input, args.output, args.target)
