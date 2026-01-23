import pandas as pd
import joblib
import os
import argparse
from sklearn.metrics import accuracy_score, classification_report
from . import settings
from . import preprocess
from . import scraper_bulk

def evaluate(start_year, end_year):
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
    # Check if raw csv exists, else scrape
    csv_path = os.path.join(settings.RAW_DATA_DIR, f"results_{start_year}_{end_year}.csv")
    if os.path.exists(csv_path):
        print(f"Loading data from {csv_path}...")
        raw_df = pd.read_csv(csv_path)
    else:
        print(f"Data not found locally. Scraping {start_year}-{end_year}...")
        # Note: bulk_scrape usually saves to specific year-year file.
        # calling it here might be complex if it's not designed to return DF directly for multiple years cleanly without saving.
        # For simplicity, we assume data exists or user runs scraper per instructions.
        # Or we call scraper logic:
        raw_df = scraper_bulk.bulk_scrape(start_year, end_year)
    
    if raw_df.empty:
        print("No data found.")
        return

    # 3. Transform (NOT Fit)
    print("Preprocessing (Transform mode)...")
    df = preprocess.transform(raw_df, artifacts)
    
    # 4. Predict
    features = [
        'jockey_win_rate', 'horse_id', 'jockey_id', 'waku', 'umaban',
        'course_type', 'distance', 'weather', 'condition',
        'lag1_rank', 'lag1_speed_index', 'interval'
    ]
    target = 'rank_class'
    
    print("Predicting...")
    preds_prob = model.predict(df[features])
    preds_label = [p.argmax() for p in preds_prob]
    
    # 5. Metrics
    if target in df.columns:
        acc = accuracy_score(df[target], preds_label)
        print(f"\nAccuracy: {acc:.4f}")
        print("\nClassification Report:")
        print(classification_report(df[target], preds_label))
        
        # 6. ROI Simulation (Simple)
        # Strategy: Bet on Top 1 Score (Score = P(Win)^4 * Odds)
        # We need "odds" column.
        
        if 'odds' in raw_df.columns:
            print("\n--- ROI Simulation (Win Bet on Top Score) ---")
            # Re-attach odds/rank from raw to processed (indices should match if no drop)
            # preprocess might drop rows? 
            # In transform, we didn't drop rows explicitly except fillna.
            # But preprocess.fit_transform drops no-rank rows.
            # transform also converts rank to numeric.
            
            sim_df = df.copy()
            sim_df['prob_win'] = preds_prob[:, 0]
            sim_df['odds'] = pd.to_numeric(raw_df['odds'], errors='coerce').fillna(0)
            sim_df['rank'] = pd.to_numeric(raw_df['rank'], errors='coerce')
            sim_df['race_id'] = raw_df['race_id']
            
            sim_df['score'] = (sim_df['prob_win'] ** 4) * sim_df['odds']
            
            # Select Top 1 per race
            top1_list = []
            grouped = sim_df.groupby('race_id')
            
            total_bet = 0
            total_return = 0
            bets_count = 0
            hits = 0
            
            for rid, group in grouped:
                if group['odds'].sum() == 0: continue # Skip races with no odds info
                
                # Pick horse with max score
                # Filter reasonable odds? (e.g. > 1.0)
                candidates = group[group['odds'] > 1.0]
                if candidates.empty: continue
                
                pick = candidates.loc[candidates['score'].idxmax()]
                
                total_bet += 100 # 100 yen per race
                bets_count += 1
                
                if pick['rank'] == 1:
                    hits += 1
                    total_return += 100 * pick['odds']
            
            if bets_count > 0:
                roi = (total_return / total_bet) * 100 
                print(f"Total Races Bet: {bets_count}")
                print(f"Hits: {hits} (Win Rate: {hits/bets_count*100:.2f}%)")
                print(f"Total Return: {total_return:.0f} yen / {total_bet:.0f} yen")
                print(f"ROI: {roi:.2f}%")
            else:
                print("No bets placed (missing odds data?).")
            
    else:
        print("Target column not found, cannot evaluate.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, default=2025)
    parser.add_argument("--end", type=int, default=2025)
    args = parser.parse_args()
    
    evaluate(args.start, args.end)
