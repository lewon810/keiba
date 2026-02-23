import pandas as pd
import joblib
import os
import argparse
from . import settings
from . import preprocess
from . import scraper_bulk
from .features import FEATURES

def evaluate(start_year, end_year, csv_file=None, min_score=None):
    print(f"--- Evaluaton Mode: {start_year}-{end_year} ---")
    
    # 1. Load Model & Artifacts
    if not os.path.exists(settings.MODEL_PATH):
        print("Model not found. Run learn.train first.")
        return {}

    print(f"Loading model from {settings.MODEL_PATH}...")
    model = joblib.load(settings.MODEL_PATH)
    encoder_path = os.path.join(settings.MODEL_DIR, 'encoders.pkl')
    artifacts = joblib.load(encoder_path)
    
    # 2. Get Data
    raw_df = pd.DataFrame()
    if csv_file and os.path.exists(csv_file):
        print(f"Loading data from provided CSV: {csv_file}...")
        raw_df = pd.read_csv(csv_file)
    else:
        # Check for individual year files first (common case)
        dfs = []
        full_range_found = True
        for y in range(start_year, end_year + 1):
            y_path = os.path.join(settings.RAW_DATA_DIR, f"results_{y}.csv")
            if os.path.exists(y_path):
                try:
                    dfs.append(pd.read_csv(y_path))
                except Exception as e:
                    print(f"Error reading {y_path}: {e}")
                    full_range_found = False
            else:
                full_range_found = False
        
        if full_range_found and dfs:
            print(f"Loading data from individual files for {start_year}-{end_year}...")
            raw_df = pd.concat(dfs, ignore_index=True)
        else:
            # Fallback to looking for a combined file (rare)
            csv_path = os.path.join(settings.RAW_DATA_DIR, f"results_{start_year}_{end_year}.csv")
            if os.path.exists(csv_path):
                print(f"Loading data from {csv_path}...")
                raw_df = pd.read_csv(csv_path)
            else:
                print(f"Data not completely found locally. Scraping {start_year}-{end_year}...")
                # scraper_bulk does not return the df, it saves to files.
                scraper_bulk.bulk_scrape(start_year, end_year)
                
                # Reload from files
                dfs = []
                for y in range(start_year, end_year + 1):
                    y_path = os.path.join(settings.RAW_DATA_DIR, f"results_{y}.csv")
                    if os.path.exists(y_path):
                        dfs.append(pd.read_csv(y_path))
                
                if dfs:
                    raw_df = pd.concat(dfs, ignore_index=True)

    if raw_df.empty:
        print("No data found.")
        return {}

    # 3. Transform (NOT Fit)
    print("Preprocessing (Transform mode)...")
    df = preprocess.transform(raw_df, artifacts)
    
    # 4. Predict
    features = FEATURES
    
    if df.empty:
        print("No data available for prediction after preprocessing.")
        return {}

    # モデルの出力を一時保存
    df['race_id'] = raw_df['race_id'].astype(str)
    df['temp_score'] = model.predict(df[features])
    
    # レース（race_id）単位で softmax を適用して win_prob を算出
    import numpy as np
    def softmax(group):
        exps = np.exp(group - group.max())
        return exps / exps.sum()
        
    df['win_prob'] = df.groupby('race_id')['temp_score'].transform(softmax)
    df = df.drop(columns=['temp_score'])
    
    # 5. Metrics (Ranking Accuracy)
    metrics = {}
    if 'rank' in raw_df.columns:
        # Attach raw info for evaluation
        df['race_id'] = raw_df['race_id']
        df['rank'] = pd.to_numeric(raw_df['rank'], errors='coerce')
        df['odds'] = pd.to_numeric(raw_df['odds'], errors='coerce').fillna(0)
        
        # スコア計算: LambdaRankスコア × オッズ
        df['score'] = df['win_prob'] * df['odds']
        
        # Metrics Initialization
        total_races = 0
        correct_top1 = 0
        total_bet = 0
        total_return = 0

        # Betting Strategy Logic — settingsから取得
        betting_type = getattr(settings, 'BETTING_TYPE', 'win')
        
        # Determine effective min_roi_score
        config_min_score = getattr(settings, 'MIN_BETTING_ROI_SCORE', 0.0)
        min_roi_score = min_score if min_score is not None else config_min_score
        
        print(f"Simulating Betting Strategy: {betting_type} (Min Score: {min_roi_score})")
        
        # Counters for races we actually bet on
        bet_races = 0
        
        grouped = df.groupby('race_id')
        for rid, group in grouped:
            if group.empty: continue
            
            # Skip if no rank=1 in group (anomaly)
            if not (group['rank'] == 1).any(): continue
            
            # total_races += 1 # This was counting all valid races in data. Moving this meaning to bet_races or keep as denominator?
            # Usually for strategy evaluation, we care about Hit Rate = Hits / Bets.
            # So I will use bet_races as the denominator for stats.
            
            # Sort by Predicted Score Descending
            # If multiple models/scores exist, ensure we use the main one.
            # Here 'score' is from model.predict
            group_sorted = group.sort_values('score', ascending=False)
            
            # Top predictions
            top1 = group_sorted.iloc[0]
            top2 = group_sorted.iloc[1] if len(group) >= 2 else None
            top3 = group_sorted.iloc[2] if len(group) >= 3 else None
            
            # Filter by Min ROI Score (Check Top1)
            if top1['score'] < min_roi_score:
                continue
            
            bet_races += 1
            
            # Actual Ranks (Horse IDs or Umaban could be used, but we use rank column on the predicted rows)
            # We need to know the actual rank of our predicted horses.
            # group_sorted contains 'rank' column from raw_df
            
            hit = False
            payout = 0
            cost = 100 # Base cost
            
            if betting_type == 'win':
                # Single Win on Top 1
                if top1['rank'] == 1:
                    hit = True
                    payout = 100 * top1['odds']
            
            elif betting_type == 'place':
                # Place bet on Top 1 (Rank 1-3)
                if top1['rank'] <= 3:
                    hit = True
                    # Cannot calc payout without place odds
                    
            elif betting_type == 'trifecta':
                # 3-Ren-Tan (Exact order 1-2-3)
                if top1['rank'] == 1 and top2 and top2['rank'] == 2 and top3 and top3['rank'] == 3:
                    hit = True
                    
            elif betting_type == 'box_trifecta':
                # 3-Ren-Tan Box (Any order of top 3 horses in top 3 ranks)
                if top1 and top2 and top3:
                    ranks = [top1['rank'], top2['rank'], top3['rank']]
                    if set(ranks) == {1, 2, 3}:
                        hit = True
                    cost = 600 # 6 combinations * 100
                    
            elif betting_type == 'uma_ren':
                # Uma-Ren (Top 2 in 1st/2nd any order)
                if top1 and top2:
                    ranks = {top1['rank'], top2['rank']}
                    if ranks == {1, 2}:
                        hit = True
                        
            elif betting_type == 'wide':
                # Wide (Top 2 both in Top 3)
                if top1 and top2:
                    if top1['rank'] <= 3 and top2['rank'] <= 3:
                        hit = True
            
            if hit:
                correct_top1 += 1 # Reusing variable as "Hits"
                total_return += payout
            
            total_bet += cost
            
        acc = correct_top1 / bet_races if bet_races > 0 else 0
        roi = (total_return / total_bet) * 100 if total_bet > 0 else 0
        
        print(f"\n--- Evaluation Result ({start_year}-{end_year}) ---")
        print(f"Strategy: {betting_type}")
        print(f"Bet Races: {bet_races} (Skipped: {len(grouped) - bet_races})")
        print(f"Hit Rate: {acc:.4f} ({correct_top1}/{bet_races})")
        if betting_type == 'win':
            print(f"ROI (Win Bet): {roi:.2f}% ({total_return:.0f}/{total_bet:.0f})")
        else:
            print(f"ROI: Cannot calculate (Missing odds for {betting_type})")

        metrics = {
            'betting_type': betting_type,
            'total_races': len(grouped),
            'bet_races': bet_races,
            'hit_rate': acc,
            'roi': roi,
            'total_return': total_return,
            'total_bet': total_bet
        }

    else:
        print("Rank column not found in raw data, cannot evaluate metrics.")
    
    return metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, default=2025)
    parser.add_argument("--end", type=int, default=2025)
    parser.add_argument("--csv", type=str, help="Path to existing CSV file")
    parser.add_argument("--min_score", type=float, help="Override min_betting_roi_score")
    args = parser.parse_args()
    
    evaluate(args.start, args.end, csv_file=args.csv, min_score=args.min_score)
