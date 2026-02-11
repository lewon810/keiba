import lightgbm as lgb
import pandas as pd
import joblib
import os
from . import settings
from . import preprocess

import argparse

def train_model(start_year, end_year):
    print(f"--- Training Mode: {start_year}-{end_year} ---")
    
    # 1. Load Data
    raw_df = preprocess.load_data(start_year=start_year, end_year=end_year)
    if raw_df.empty:
        print("No training data found.")
        return

    # 2. Preprocess
    # Now returns df AND artifacts (encoders, maps)
    df, artifacts = preprocess.preprocess(raw_df)
    
    # Clean numeric columns (just in case)
    df['waku'] = pd.to_numeric(df['waku'], errors='coerce').fillna(0)
    df['umaban'] = pd.to_numeric(df['umaban'], errors='coerce').fillna(0)

    # Split
    train, valid, _ = preprocess.split_data(df)
    
    # 3. Train with LambdaRank
    features = [
        'jockey_win_rate', 'trainer_win_rate', 'horse_id', 'jockey_id', 'trainer_id',
        'waku', 'umaban', 'course_type', 'distance', 'weather', 'condition',
        'lag1_rank', 'lag1_speed_index', 'lag1_last_3f', 'interval', 'weight_diff',
        'sire_id', 'damsire_id', 'running_style',
        'sire_win_rate', 'damsire_win_rate',
        'course_type_win_rate', 'dist_cat_win_rate',
        'front_runner_count', 'pace_ratio'
    ]
    target = 'rank'  # Changed from 'rank_class' to 'rank' for LambdaRank
    
    print(f"Features: {features}")
    
    # Sort by race_id to ensure group parameter alignment
    train = train.sort_values('race_id').reset_index(drop=True)
    valid = valid.sort_values('race_id').reset_index(drop=True)
    
    # Create group parameter (number of horses per race)
    train_groups = train.groupby('race_id').size().to_list()
    valid_groups = valid.groupby('race_id').size().to_list()
    
    print(f"Train: {len(train)} rows, {len(train_groups)} races")
    print(f"Valid: {len(valid)} rows, {len(valid_groups)} races")
    print(f"Train group sum: {sum(train_groups)}, should match {len(train)}")
    print(f"Valid group sum: {sum(valid_groups)}, should match {len(valid)}")
    
    # Verify group parameter consistency
    assert sum(train_groups) == len(train), "Train group parameter mismatch!"
    assert sum(valid_groups) == len(valid), "Valid group parameter mismatch!"
    
    lgb_train = lgb.Dataset(train[features], train[target], group=train_groups)
    lgb_eval = lgb.Dataset(valid[features], valid[target], group=valid_groups, reference=lgb_train)
    
    params = {
        'objective': 'lambdarank',
        'metric': 'ndcg',
        'ndcg_eval_at': [1, 3, 5],
        'boosting_type': 'gbdt',
        'learning_rate': 0.05,
        'num_leaves': 31,
        'verbose': -1,
        'seed': 42
    }
    
    print("Starting LambdaRank training...")
    model = lgb.train(
        params,
        lgb_train,
        valid_sets=[lgb_train, lgb_eval],
        num_boost_round=1000, # Increased rounds
        callbacks=[
            lgb.early_stopping(stopping_rounds=20),
            lgb.log_evaluation(50)
        ]
    )
    
    # Save Model
    os.makedirs(settings.MODEL_DIR, exist_ok=True)
    joblib.dump(model, settings.MODEL_PATH)
    
    # Calculate Feature Importance (Gain)
    importance = model.feature_importance(importance_type='gain')
    feature_importance = pd.DataFrame({
        'feature': features,
        'importance': importance
    }).sort_values('importance', ascending=False)
    
    print("\nFeature Importance (Gain):")
    print(feature_importance.head(20))
    
    # Save Encoders & Importance to Artifacts
    artifacts['feature_importance'] = feature_importance.to_dict('records')
    
    encoder_path = os.path.join(settings.MODEL_DIR, 'encoders.pkl')
    joblib.dump(artifacts, encoder_path)
    print(f"Model saved to {settings.MODEL_PATH}")
    print(f"Artifacts (Encoders + Importance) saved to {encoder_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, default=2016)
    parser.add_argument("--end", type=int, default=2024)
    args = parser.parse_args()
    
    train_model(args.start, args.end)
