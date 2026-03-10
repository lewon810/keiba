import lightgbm as lgb
import pandas as pd
import joblib
import os
from . import settings
from . import preprocess
from .features import FEATURES

import argparse

def train_model(start_year, end_year, start_month=None, end_month=None):
    if start_month and end_month:
        print(f"--- Training Mode: {start_year}/{start_month}-{end_year}/{end_month} ---")
    else:
        print(f"--- Training Mode: {start_year}-{end_year} ---")
    
    # 1. Load Data
    raw_df = preprocess.load_data(start_year=start_year, end_year=end_year, start_month=start_month, end_month=end_month)
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
    features = FEATURES
    target = 'rank_class'  # 0=1着, 1=2-3着, 2=4-5着, 3=6着以降
    # rank（着順）ではなく rank_class を使用する理由:
    # rank は popularity と強相関（0.59）のため、モデルが人気予測器になってしまう。
    # rank_class は4クラスの粗い分類なので過剰な順位学習を防ぐ。
    
    print(f"Features ({len(features)}): {features}")
    
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
        'learning_rate': 0.05,          # 0.03→0.05（収束速度向上）
        'num_leaves': 127,              # 63→127（より複雑なパターンを学習）
        'min_child_samples': 30,        # 50→30（小さなグループも考慮）
        'feature_fraction': 0.8,        # 特徴量サブサンプリング
        'bagging_fraction': 0.8,        # データサブサンプリング
        'bagging_freq': 5,
        'lambda_l1': 0.1,              # L1正則化
        'lambda_l2': 0.5,              # L2正則化（0.03→0.5，少し緩和）
        'max_depth': 8,
        'label_gain': [15, 2, 1, 0],   # rank_class=0(1着)にGAIN=15, =1(2-3着)=2, =2(4-5着)=1, =3(6着+)=0
        'verbose': -1,
        'seed': 42
    }
    
    print("Starting LambdaRank training...")
    model = lgb.train(
        params,
        lgb_train,
        valid_sets=[lgb_train, lgb_eval],
        num_boost_round=5000,  # 2000→5000（データ70万件、early stoppingで制御）
        callbacks=[
            lgb.early_stopping(stopping_rounds=50),  # 20→50（最適解を広く探索）
            lgb.log_evaluation(100)
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
    parser.add_argument("--start_month", type=int, default=None)
    parser.add_argument("--end_month", type=int, default=None)
    args = parser.parse_args()
    
    train_model(args.start, args.end, args.start_month, args.end_month)
