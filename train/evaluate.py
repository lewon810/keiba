import pandas as pd
import joblib
import os
import argparse
from . import settings
from . import preprocess
from . import scraper_bulk

def evaluate(start_year, end_year, csv_file=None):
    print(f"--- Evaluaton Mode: {start_year}-{end_year} ---")
    
    # 1. Load Model & Artifacts
    if not os.path.exists(settings.MODEL_PATH):
        print("Model not found. Run learn.train first.")
        return

    print(f"Loading model from {settings.MODEL_PATH}...")
    model = joblib.load(settings.MODEL_PATH)
    encoder_path = os.path.join(settings.MODEL_DIR, 'encoders.pkl')
    artifacts = joblib.load(encoder_path)
    
    # 2. Get Data
    if csv_file and os.path.exists(csv_file):
        print(f"Loading data from provided CSV: {csv_file}...")
        raw_df = pd.read_csv(csv_file)
    else:
        # Check if raw csv exists based on years
        csv_path = os.path.join(settings.RAW_DATA_DIR, f"results_{start_year}_{end_year}.csv")
        if os.path.exists(csv_path):
            print(f"Loading data from {csv_path}...")
            raw_df = pd.read_csv(csv_path)
        else:
            print(f"Data not found locally. Scraping {start_year}-{end_year}...")
            raw_df = scraper_bulk.bulk_scrape(start_year, end_year)
    
    if raw_df.empty:
        print("No data found.")
        return

    # --- Filtering based on betting.yaml ---
    import yaml
    yaml_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'betting.yaml')
    if os.path.exists(yaml_path):
        print(f"Loading filters from {yaml_path}...")
        with open(yaml_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
            
        if config:
            # Filter by Place Code
            # Race ID: YYYY PP K D RR (String)
            # Ensure race_id is string
            raw_df['race_id'] = raw_df['race_id'].astype(str)
            
            target_places = config.get('target_places', [])
            if target_places:
                # Place code is index 4,5 (0-based) -> chars at 4:6? 
                # e.g. 2023 06 ... -> 2023 is 0:4, 06 is 4:6
                print(f"Filtering for places: {target_places}")
                # Extract place code
                raw_df['place_code'] = raw_df['race_id'].str[4:6]
                raw_df = raw_df[raw_df['place_code'].isin([str(p).zfill(2) for p in target_places])]
                print(f"  Rows after place filter: {len(raw_df)}")

            # Filter by Race Number
            target_races = config.get('target_race_numbers', [])
            if target_races:
                # Last 2 digits? usually.
                print(f"Filtering for race numbers: {target_races}")
                raw_df['race_no'] = raw_df['race_id'].str[-2:].astype(int)
                raw_df = raw_df[raw_df['race_no'].isin(target_races)]
                print(f"  Rows after race_no filter: {len(raw_df)}")

    # 3. Transform (NOT Fit)
    print("Preprocessing (Transform mode)...")
    df = preprocess.transform(raw_df, artifacts)
    
    # 4. Predict
    features = [
        'jockey_win_rate', 'horse_id', 'jockey_id', 'waku', 'umaban',
        'course_type', 'distance', 'weather', 'condition',
        'lag1_rank', 'lag1_speed_index', 'interval'
    ]
    
    print("Predicting...")
    # LightGBM Ranker returns 1D raw scores
    scores = model.predict(df[features])
    df['score'] = scores
    
    # 5. Metrics (Ranking Accuracy)
    if 'rank' in raw_df.columns:
        # Attach raw info for evaluation
        df['race_id'] = raw_df['race_id']
        df['rank'] = pd.to_numeric(raw_df['rank'], errors='coerce')
        df['odds'] = pd.to_numeric(raw_df['odds'], errors='coerce').fillna(0)
        
        # Group by race -> find horse with max score -> check if rank==1
        total_races = 0
        correct_top1 = 0
        
        # ROI Stats
        total_bet = 0
        total_return = 0
        
        grouped = df.groupby('race_id')
        for rid, group in grouped:
            if group.empty: continue
            
            # Skip if no rank=1 in group (anomaly)
            if not (group['rank'] == 1).any(): continue
            
            total_races += 1
            
            # Predicted Winner
            if group['score'].nunique() == 1:
                 pred_winner = group.iloc[0] 
            else:
                 pred_winner = group.loc[group['score'].idxmax()]
            
            # Check accuracy
            if pred_winner['rank'] == 1:
                correct_top1 += 1
                
                # ROI (Bet 100 yen on Win)
                if pred_winner['odds'] > 0:
                    total_return += 100 * pred_winner['odds']
            
            # Always bet 100
            total_bet += 100
            
        acc = correct_top1 / total_races if total_races > 0 else 0
        roi = (total_return / total_bet) * 100 if total_bet > 0 else 0
        
        print(f"\n--- Evaluation Result ({start_year}-{end_year}) ---")
        print(f"Total Races: {total_races}")
        print(f"Accuracy@1 (Winner predicted correctly): {acc:.4f} ({correct_top1}/{total_races})")
        print(f"ROI (Win Bet): {roi:.2f}% ({total_return:.0f}/{total_bet:.0f})")

    else:
        print("Rank column not found in raw data, cannot evaluate metrics.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, default=2025)
    parser.add_argument("--end", type=int, default=2025)
    parser.add_argument("--csv", type=str, help="Path to existing CSV file")
    args = parser.parse_args()
    
    evaluate(args.start, args.end, csv_file=args.csv)
