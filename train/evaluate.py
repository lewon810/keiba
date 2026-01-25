import pandas as pd
import joblib
import os
import argparse
from . import settings
from . import preprocess
from . import scraper_bulk

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
        return {}

    # --- Filtering based on evaluate_settings.yml ---
    import yaml
    yaml_path = os.path.join(os.path.dirname(__file__), 'evaluate_settings.yml')
    config = {}
    
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
    
    if df.empty:
        print("No data available for prediction after preprocessing.")
        return {}

    print("Predicting...")
    # LightGBM Multiclass returns (N, 4) probability matrix
    # Class 0 is Rank 1 (Winner)
    pred_probs = model.predict(df[features])
    
    # Use Probability of Winning (Class 0) as base score
    # Handle both 1D (binary) and 2D (multiclass) outputs just in case
    if pred_probs.ndim > 1:
        df['win_prob'] = pred_probs[:, 0]
    else:
        df['win_prob'] = pred_probs
    
    # 5. Metrics (Ranking Accuracy)
    metrics = {}
    if 'rank' in raw_df.columns:
        # Attach raw info for evaluation
        df['race_id'] = raw_df['race_id']
        df['rank'] = pd.to_numeric(raw_df['rank'], errors='coerce')
        df['odds'] = pd.to_numeric(raw_df['odds'], errors='coerce').fillna(0)
        
        # Calculate Expectation Score: (Win Prob)^4 * Odds
        # This highlights horses with high win probability AND decent return
        df['score'] = (df['win_prob'] ** 4) * df['odds']
        
        # Metrics Initialization
        total_races = 0
        correct_top1 = 0
        total_bet = 0
        total_return = 0

        # Betting Strategy Logic
        betting_type = config.get('betting_type', 'win')
        
        # Determine effective min_roi_score
        config_min_score = config.get('min_betting_roi_score', 0.0)
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
