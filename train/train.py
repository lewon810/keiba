import lightgbm as lgb
import pandas as pd
import joblib
import os
from sklearn.metrics import accuracy_score
from . import settings
from . import preprocess

def train_model():
    # 1. Load Data
    raw_df = preprocess.load_data()
    print("Data loaded.")

    # 2. Preprocess
    # Now returns df AND artifacts (encoders, maps)
    df, artifacts = preprocess.preprocess(raw_df)
    
    # Clean numeric columns (just in case)
    df['waku'] = pd.to_numeric(df['waku'], errors='coerce').fillna(0)
    df['umaban'] = pd.to_numeric(df['umaban'], errors='coerce').fillna(0)

    # Split
    train, valid, test = preprocess.split_data(df)
    
    # 3. Train
    features = [
        'jockey_win_rate', 'horse_id', 'jockey_id', 'waku', 'umaban',
        'course_type', 'distance', 'weather', 'condition',
        'lag1_rank', 'lag1_speed_index', 'interval'
    ]
    target = 'rank_class'
    
    print(f"Features: {features}")
    
    lgb_train = lgb.Dataset(train[features], train[target])
    lgb_eval = lgb.Dataset(valid[features], valid[target], reference=lgb_train)
    
    params = {
        'objective': 'multiclass',
        'num_class': 4,
        'metric': 'multi_logloss',
        'boosting_type': 'gbdt',
        'learning_rate': 0.05,
        'num_leaves': 31,
        'verbose': -1,
        'seed': 42
    }
    
    print("Starting training...")
    model = lgb.train(
        params,
        lgb_train,
        valid_sets=[lgb_train, lgb_eval],
        num_boost_round=100,
        callbacks=[
            lgb.early_stopping(stopping_rounds=10),
            lgb.log_evaluation(10)
        ]
    )
    
    # Save Model
    os.makedirs(settings.MODEL_DIR, exist_ok=True)
    joblib.dump(model, settings.MODEL_PATH)
    
    # Save Encoders
    encoder_path = os.path.join(settings.MODEL_DIR, 'encoders.pkl')
    joblib.dump(artifacts, encoder_path)
    print(f"Model saved to {settings.MODEL_PATH}")
    print(f"Encoders saved to {encoder_path}")

if __name__ == "__main__":
    train_model()
