import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import os
import joblib
import numpy as np

# Config
DATA_PATH = r'c:\Users\lewon\keiba\learn\data\raw\results_2016_2025.csv'
MODEL_DIR = r'c:\Users\lewon\keiba\learn\models'
os.makedirs(MODEL_DIR, exist_ok=True)

def load_data():
    print(f"Loading data from {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH)
    return df

def feature_engineering(df):
    print("Feature Engineering...")
    
    # Sort by race_id to ensure past data comes before future data
    df = df.sort_values('race_id')
    
    # Convert rank to numeric
    df['rank_num'] = pd.to_numeric(df['rank'], errors='coerce')
    # Create binary target for win (1st place)
    df['is_win'] = (df['rank_num'] == 1).astype(int)
    
    # --- Historical Statistics ---
    
    # Function to calculate expanding win rate
    def calculate_rolling_stats(df, group_col, target_col='is_win'):
        # Group by entity and calculate expanding sum/count
        # shift(1) ensures we don't use the current race result
        g = df.groupby(group_col)[target_col]
        
        wins = g.expanding().sum().shift(1).reset_index(level=0, drop=True)
        counts = g.expanding().count().shift(1).reset_index(level=0, drop=True)
        
        return wins, counts

    # Horse Stats
    print("Calculating Horse Stats...")
    df['horse_wins'], df['horse_races'] = calculate_rolling_stats(df, 'horse_id')
    df['horse_win_rate'] = df['horse_wins'] / df['horse_races']
    df['horse_win_rate'] = df['horse_win_rate'].fillna(0) # First race or no history
    df['horse_races'] = df['horse_races'].fillna(0)

    # Jockey Stats
    print("Calculating Jockey Stats...")
    df['jockey_wins'], df['jockey_races'] = calculate_rolling_stats(df, 'jockey_id')
    df['jockey_win_rate'] = df['jockey_wins'] / df['jockey_races']
    df['jockey_win_rate'] = df['jockey_win_rate'].fillna(0)
    df['jockey_races'] = df['jockey_races'].fillna(0)
    
    return df

def preprocess(df):
    print("Preprocessing...")
    
    # Drop rows where we can't evaluate rank (for training/eval), 
    # BUT for feature engineering we arguably should have kept them? 
    # Actually, expanding stats work better if we have all data. 
    # So filter AFTER feature engineering.
    
    # Filter valid ranks for training
    df = df[df['rank_num'].notnull()].copy()
    df['rank'] = df['rank_num'].astype(int)
    
    # Drop temp cols
    cols_to_drop = ['time', 'odds', 'popularity', 'horse_name', 'jockey', 'date', 'rank_num', 'is_win', 
                    'horse_wins', 'jockey_wins'] # Keep rates and counts
    
    # Features
    cat_cols = ['course_type', 'weather', 'condition', 'horse_id', 'jockey_id', 'distance']
    num_cols = ['waku', 'umaban', 'horse_win_rate', 'horse_races', 'jockey_win_rate', 'jockey_races']
    
    # Handle distance as categorical
    df['distance'] = df['distance'].astype(str)

    # Label Encoding
    encoders = {}
    for col in cat_cols:
        df[col] = df[col].astype(str)
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le
    
    # Save encoders
    joblib.dump(encoders, os.path.join(MODEL_DIR, 'encoders.pkl'))
    
    # Keep race_id for grouping
    return df, encoders, cat_cols, num_cols

def train_model(df, cat_cols, num_cols):
    print("Training LightGBM with Historical Features...")
    
    # Prepare X, y
    feature_cols = cat_cols + num_cols
    X = df[feature_cols]
    y = df['rank']
    groups = df['race_id']
    
    # Split (Time-series like split on race_ids)
    unique_race_ids = df['race_id'].unique()
    train_size = int(len(unique_race_ids) * 0.8)
    train_race_ids = unique_race_ids[:train_size]
    val_race_ids = unique_race_ids[train_size:]
    
    train_mask = df['race_id'].isin(train_race_ids)
    val_mask = df['race_id'].isin(val_race_ids)
    
    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]
    
    # Groups for Ranker
    # IMPORTANT: Data must be sorted by group for most LGBM implementations, or at least grouped together.
    # df is already sorted by race_id.
    group_train = df[train_mask].groupby('race_id').size().to_numpy()
    group_val = df[val_mask].groupby('race_id').size().to_numpy()

    print(f"Train Shape: {X_train.shape}, Val Shape: {X_val.shape}")

    # Define model
    model = lgb.LGBMRanker(
        objective='lambdarank',
        metric='ndcg',
        n_estimators=1000,
        learning_rate=0.03, # Lower LR for better generalization with more features
        num_leaves=31,
        random_state=42,
        importance_type='gain'
    )
    
    # Train
    model.fit(
        X_train, y_train,
        group=group_train,
        eval_set=[(X_val, y_val)],
        eval_group=[group_val],
        eval_at=[1], 
        callbacks=[lgb.early_stopping(stopping_rounds=100), lgb.log_evaluation(50)],
        categorical_feature=cat_cols
    )
    
    # Save model
    joblib.dump(model, os.path.join(MODEL_DIR, 'lgbm_ranker_v2.pkl'))
    print(f"Model saved to {MODEL_DIR}")
    
    # Feature Importance
    print("\nFeature Importances:")
    importances = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    print(importances)
    
    return model

if __name__ == "__main__":
    df = load_data()
    # Apply feature engineering BEFORE filtering/preprocessing
    df_engineered = feature_engineering(df)
    df_processed, encoders, cat_cols, num_cols = preprocess(df_engineered)
    model = train_model(df_processed, cat_cols, num_cols)
