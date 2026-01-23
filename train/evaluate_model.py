import pandas as pd
import lightgbm as lgb
import os
import joblib
import numpy as np

# Config
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # keiba/
DATA_PATH = os.path.join(BASE_DIR, 'train', 'data', 'raw', 'results_validation_patched.csv')
MODEL_PATH = os.path.join(BASE_DIR, 'train', 'models', 'lgbm_ranker_v2.pkl')
ENCODERS_PATH = os.path.join(BASE_DIR, 'train', 'models', 'encoders.pkl')

def load_data():
    if not os.path.exists(DATA_PATH):
        print(f"Data not found at {DATA_PATH}")
        return pd.DataFrame() # Return empty to avoid crash, but evaluate handles check
    print(f"Loading data from {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH)
    return df

def feature_engineering(df):
    if df.empty: return df
    # Sort by race_id
    df = df.sort_values('race_id')
    
    # Convert rank
    df['rank_num'] = pd.to_numeric(df['rank'], errors='coerce')
    df['is_win'] = (df['rank_num'] == 1).astype(int)
    
    # --- Historical Statistics (Same as training) ---
    def calculate_rolling_stats(df, group_col, target_col='is_win'):
        g = df.groupby(group_col)[target_col]
        wins = g.expanding().sum().shift(1).reset_index(level=0, drop=True)
        counts = g.expanding().count().shift(1).reset_index(level=0, drop=True)
        return wins, counts

    df['horse_wins'], df['horse_races'] = calculate_rolling_stats(df, 'horse_id')
    df['horse_win_rate'] = (df['horse_wins'] / df['horse_races']).fillna(0)
    df['horse_races'] = df['horse_races'].fillna(0)

    df['jockey_wins'], df['jockey_races'] = calculate_rolling_stats(df, 'jockey_id')
    df['jockey_win_rate'] = (df['jockey_wins'] / df['jockey_races']).fillna(0)
    df['jockey_races'] = df['jockey_races'].fillna(0)
    
    return df

def preprocess(df, encoders):
    if df.empty: return df, []
    
    # Filter valid ranks for evaluation reference (though for inference we typically don't know rank, 
    # but this is 'evaluate_model', so we are evaluating on labeled data)
    df = df[df['rank_num'].notnull()].copy()
    df['rank'] = df['rank_num'].astype(int)
    
    # Features
    cat_cols = ['course_type', 'weather', 'condition', 'horse_id', 'jockey_id', 'distance']
    num_cols = ['waku', 'umaban', 'horse_win_rate', 'horse_races', 'jockey_win_rate', 'jockey_races']
    
    df['distance'] = df['distance'].astype(str)

    # Encode
    for col in cat_cols:
        df[col] = df[col].astype(str)
        le = encoders.get(col)
        if le:
            # Safe transform
            mapping = dict(zip(le.classes_, le.transform(le.classes_)))
            df[col] = df[col].map(mapping).fillna(-1).astype(int)
            
    return df, cat_cols + num_cols

def evaluate():
    if not os.path.exists(DATA_PATH):
        print(f"Patched data not found at {DATA_PATH}. Please run patch_validation_odds.py first.")
        return

    df = load_data()
    if df.empty: return
    
    df_eng = feature_engineering(df)
    
    encoders = joblib.load(ENCODERS_PATH)
    df_processed, feature_cols = preprocess(df_eng, encoders)
    
    # Split same as training to get Validation Set
    unique_race_ids = df_processed['race_id'].unique()
    train_size = int(len(unique_race_ids) * 0.8)
    val_race_ids = unique_race_ids[train_size:]
    
    val_df = df_processed[df_processed['race_id'].isin(val_race_ids)].copy()
    
    print(f"Validation Data: {len(val_df)} rows, {len(val_df['race_id'].unique())} races")
    
    X_val = val_df[feature_cols]
    
    model = joblib.load(MODEL_PATH)
    
    # Predict
    print("Predicting...")
    # LGBM Ranker returns raw scores (higher is better)
    val_df['score'] = model.predict(X_val)
    
    # Calculate Metrics
    total_races = 0
    hits_1 = 0
    hits_3 = 0
    
    total_cost = 0
    total_return = 0
    
    bet_amount = 100
    
    for race_id, group in val_df.groupby('race_id'):
        total_races += 1
        
        # Sort by predicted score
        group = group.sort_values('score', ascending=False)
        pred_winner = group.iloc[0]
        
        # Accuracy
        if pred_winner['rank'] == 1:
            hits_1 += 1
            # Return
            try:
                # odds column might be object or float
                odds_val = float(pred_winner['odds']) if pd.notnull(pred_winner['odds']) else 0.0
                total_return += bet_amount * odds_val
            except:
                pass 
            
        if pred_winner['rank'] <= 3:
            hits_3 += 1
            
        total_cost += bet_amount
            
    acc_1 = hits_1 / total_races if total_races > 0 else 0
    acc_3 = hits_3 / total_races if total_races > 0 else 0
    roi = (total_return / total_cost) * 100 if total_cost > 0 else 0
    
    print(f"\nResults (Validation Set - {total_races} races):")
    print(f"Accuracy (Hit Rate): {acc_1:.2%}")
    print(f"Accuracy (Top 3): {acc_3:.2%}")
    print(f"ROI (Recovery Rate): {roi:.2f}%")

if __name__ == "__main__":
    evaluate()
