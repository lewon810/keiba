import pandas as pd
import numpy as np
import os
from . import settings

def load_data():
    """Loads all CSVs from raw data directory."""
    files = [f for f in os.listdir(settings.RAW_DATA_DIR) if f.endswith('.csv')]
    dfs = []
    for f in files:
        path = os.path.join(settings.RAW_DATA_DIR, f)
        try:
            df = pd.read_csv(path)
            dfs.append(df)
        except Exception as e:
            print(f"Skipping {f}: {e}")
            
    if not dfs:
        raise ValueError("No data found in raw directory.")
        
    return pd.concat(dfs, ignore_index=True)

def preprocess(df):
    """
    Cleaning and Feature Engineering.
    """
    print("Preprocessing data...")
    
    # Clean Rank
    df['rank'] = pd.to_numeric(df['rank'], errors='coerce')
    df = df.dropna(subset=['rank']) # Drop non-numeric ranks (e.g., "DNS", "DQ")
    
    # Create Target: rank_class
    # 0: 1st, 1: 2-3, 2: 4-5, 3: 6+
    conditions = [
        (df['rank'] == 1),
        (df['rank'] <= 3),
        (df['rank'] <= 5)
    ]
    choices = [0, 1, 2]
    df['rank_class'] = np.select(conditions, choices, default=3)
    
    # Parse Date
    df['date'] = pd.to_datetime(df['date'], format='%Y年%m月%d日', errors='coerce')
    
    # Feature: Time (seconds)
    # Format 1:34.5 -> 94.5
    def parse_time(t_str):
        try:
            if ':' in str(t_str):
                m, s = t_str.split(':')
                return int(m) * 60 + float(s)
            return float(t_str)
        except:
            return np.nan
            
    df['time_sec'] = df['time'].apply(parse_time)
    
    # Feature: Speed Index (Z-score by Course & Distance)
    # Group by CourseType + Distance
    # Note: 'course_type' and 'distance' must exist from scraper update
    if 'course_type' in df.columns and 'distance' in df.columns:
        # Filter outliers or valid times
        valid_times = df[df['time_sec'] > 0]
        
        # Calculate stats
        course_stats = valid_times.groupby(['course_type', 'distance'])['time_sec'].agg(['mean', 'std']).reset_index()
        course_stats.columns = ['course_type', 'distance', 'course_mean', 'course_std']
        
        # Merge stats
        df = df.merge(course_stats, on=['course_type', 'distance'], how='left')
        
        # Calculate deviation (Z-score), inverted so higher is faster
        # Avoid div by zero
        df['speed_index'] = (df['course_mean'] - df['time_sec']) / df['course_std'].replace(0, 1)
        df['speed_index'] = df['speed_index'].fillna(0)
    else:
        df['speed_index'] = 0

    # Feature: Lag Features (Past Performance)
    # Sort by Horse and Date
    df = df.sort_values(['horse_id', 'date'])
    
    # Lag 1: Previous Rank
    df['lag1_rank'] = df.groupby('horse_id')['rank'].shift(1).fillna(99) # Default to 99 (unranked/debut)
    
    # Lag 1: Previous Speed Index
    df['lag1_speed_index'] = df.groupby('horse_id')['speed_index'].shift(1).fillna(0)
    
    # Lag 1: Interval (Days since last race)
    df['interval'] = (df['date'] - df.groupby('horse_id')['date'].shift(1)).dt.days.fillna(365) # Default 1 year

    # Target Encoding (Jockey)
    # Fit on all data for demo
    jockey_stats = df.groupby('jockey_id')['rank'].agg(['count', lambda x: (x==1).sum()])
    jockey_stats.columns = ['count', 'wins']
    jockey_win_rate = (jockey_stats['wins'] / jockey_stats['count']).to_dict()
    
    df['jockey_win_rate'] = df['jockey_id'].map(jockey_win_rate).fillna(0)
    
    # Store artifacts
    from sklearn.preprocessing import LabelEncoder
    artifacts = {
        'jockey_win_rate': jockey_win_rate
    }

    # Encode IDs
    for col in settings.CATEGORY_COLS:
        if col in df.columns:
            # Add string conversion for safety
            df[col] = df[col].astype(str).fillna("unknown")
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            artifacts[f'le_{col}'] = le

    # Fill NaNs
    df = df.fillna(0)
    
    return df, artifacts

def transform(df, artifacts):
    """
    Apply preprocessing using existing artifacts (Encoders, Maps).
    Used for Inference and Evaluation on new data.
    """
    # Parse Date
    df['date'] = pd.to_datetime(df['date'], format='%Y年%m月%d日', errors='coerce')
    
    # Feature: Time (seconds)
    def parse_time(t_str):
        try:
            if ':' in str(t_str):
                m, s = t_str.split(':')
                return int(m) * 60 + float(s)
            return float(t_str)
        except:
            return np.nan
    df['time_sec'] = df['time'].apply(parse_time)
    
    # Feature: Speed Index (Simplified for Inference - use Pre-computed Course Stats if available)
    # Ideally, artifacts should contain course_stats. 
    # For MVP, we'll re-calculate if batch, or skip if single row/no context data.
    # In rigorous ML, stats should be fixed from Train set.
    # For now, we will calculate SI based on the current batch's deviations if possible, 
    # OR better: artifacts should save the 'course_mean/std' map.
    # Let's assume for this step we skip complex SI if stats not in artifacts, 
    # or just calc on the fly if it's a batch evaluation.
    
    # If we are valid/eval set, we might want to use our own distribution or training distribution.
    # To keep consistent with current logic which calcs SI on the input dataframe:
    if 'course_type' in df.columns and 'distance' in df.columns:
         valid_times = df[df['time_sec'] > 0]
         if not valid_times.empty:
             stats = valid_times.groupby(['course_type', 'distance'])['time_sec'].agg(['mean', 'std']).reset_index()
             stats.columns = ['course_type', 'distance', 'course_mean', 'course_std']
             df = df.merge(stats, on=['course_type', 'distance'], how='left')
             df['speed_index'] = (df['course_mean'] - df['time_sec']) / df['course_std'].replace(0, 1)
             df['speed_index'] = df['speed_index'].fillna(0)
         else:
             df['speed_index'] = 0
    else:
        df['speed_index'] = 0

    # Lag Features (Past Performance) - Self-contained sort
    df = df.sort_values(['horse_id', 'date'])
    df['lag1_rank'] = df.groupby('horse_id')['rank'].shift(1).fillna(99)
    df['lag1_speed_index'] = df.groupby('horse_id')['speed_index'].shift(1).fillna(0)
    df['interval'] = (df['date'] - df.groupby('horse_id')['date'].shift(1)).dt.days.fillna(365)

    # Encoding using Artifacts
    if 'jockey_win_rate' in artifacts:
        jockey_map = artifacts['jockey_win_rate']
        # Map with type safety fallback
        def get_rate(jid):
            if jid in jockey_map: return jockey_map[jid]
            # Try str/int conversions
            if str(jid) in jockey_map: return jockey_map[str(jid)]
            return 0.0
        df['jockey_win_rate'] = df['jockey_id'].apply(get_rate)
    else:
        df['jockey_win_rate'] = 0

    # Label Encoders
    for col in settings.CATEGORY_COLS:
        if col in df.columns:
            le_key = f'le_{col}'
            if le_key in artifacts:
                le = artifacts[le_key]
                valid_classes = set(le.classes_)
                # Handle unknown
                df[col] = df[col].astype(str).map(lambda x: x if x in valid_classes else "unknown")
                if "unknown" not in valid_classes:
                    # Fallback to 0 index class if "unknown" not explicitly trained
                    fallback = list(valid_classes)[0]
                    df[col] = df[col].map(lambda x: x if x in valid_classes else fallback)
                
                df[col] = le.transform(df[col])

    # Rank Class (for evaluation if rank exists)
    if 'rank' in df.columns:
        df['rank'] = pd.to_numeric(df['rank'], errors='coerce')
        conditions = [(df['rank'] == 1), (df['rank'] <= 3), (df['rank'] <= 5)]
        choices = [0, 1, 2]
        df['rank_class'] = np.select(conditions, choices, default=3)
    
    df = df.fillna(0)
    return df

def split_data(df):
    """
    Split into Train/Valid/Test (Time Series Split).
    """
    # Sort by date
    df = df.sort_values('date')
    
    # Simple time split
    n = len(df)
    train_idx = int(n * 0.7)
    valid_idx = int(n * 0.85)
    
    train = df.iloc[:train_idx]
    valid = df.iloc[train_idx:valid_idx]
    test = df.iloc[valid_idx:]
    
    return train, valid, test
