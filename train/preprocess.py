import pandas as pd
import numpy as np
import os
from . import settings

def load_data(start_year=None, end_year=None):
    """Loads all result CSVs from raw data directory, optionally filtering by year."""
    # Ensure we only load results_*.csv files, excluding things like horse_profiles.csv
    files = [f for f in os.listdir(settings.RAW_DATA_DIR) if f.startswith('results_') and f.endswith('.csv')]
    dfs = []
    
    # Filter files based on year in filename (results_YYYY.csv)
    target_files = []
    if start_year and end_year:
        target_years = range(start_year, end_year + 1)
        for f in files:
            # Extract year assume format "results_2024.csv"
            try:
                # Simplistic extraction
                parts = f.replace('results_', '').replace('.csv', '')
                if '_' in parts: # results_2020_2021.csv case? usually just results_YYYY
                     # If scraper_bulk outputs range? default scraper outputs results_YYYY.csv
                     # Let's rely on checking file year if possible
                     y = int(parts)
                     if y in target_years:
                         target_files.append(f)
                else:
                     y = int(parts)
                     if y in target_years:
                         target_files.append(f)
            except:
                pass
    else:
        target_files = files

    print(f"Loading data from: {target_files}")

    for f in target_files:
        path = os.path.join(settings.RAW_DATA_DIR, f)
        try:
            df = pd.read_csv(path)
            dfs.append(df)
        except Exception as e:
            print(f"Skipping {f}: {e}")
            
    if not dfs:
        # Fallback or raise
        print("No matching data found.")
        return pd.DataFrame()
        
    df = pd.concat(dfs, ignore_index=True)
    
    # Drop duplicates
    initial_len = len(df)
    df = df.drop_duplicates(subset=['race_id', 'horse_id'])
    if len(df) < initial_len:
        print(f"Dropped {initial_len - len(df)} duplicate rows.")
    
    # Merge Pedigree Data (Horse Profiles)
    profile_path = os.path.join(settings.RAW_DATA_DIR, "horse_profiles.csv")
    if os.path.exists(profile_path):
        print("Merging horse profiles (Pedigree)...")
        try:
            profiles = pd.read_csv(profile_path)
            # Ensure ID is string
            if 'horse_id' in profiles.columns:
                profiles['horse_id'] = profiles['horse_id'].astype(str)
                # Keep relevant columns
                cols_to_merge = ['horse_id', 'sire_id', 'damsire_id']
                profiles = profiles[[c for c in cols_to_merge if c in profiles.columns]]
                
                df['horse_id'] = df['horse_id'].astype(str)
                df = df.merge(profiles, on='horse_id', how='left')
                
                # Fill missing
                if 'sire_id' in df.columns:
                    df['sire_id'] = df['sire_id'].fillna("unknown")
                if 'damsire_id' in df.columns:
                    df['damsire_id'] = df['damsire_id'].fillna("unknown")
            else:
                print("Profile data missing horse_id column.")
        except Exception as e:
            print(f"Error merging profiles: {e}")
    else:
        print("No horse profile data found. Skipping pedigree features.")
        
    return df

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

    # Target Encoding (Jockey) - Expanding Window (Leakage Free)
    # Sort by date first (already done above)
    print("Calculating expanding window stats for Jockey Win Rate...")
    
    # 1. Calculate boolean 'is_win'
    df['is_win'] = (df['rank'] == 1).astype(int)
    
    # 2. Group by Jockey and calc expanding mean, shifted by 1
    # This ensures row N uses info from 0 to N-1
    # fillna(0) for the first race of a jockey
    df['jockey_win_rate'] = df.groupby('jockey_id')['is_win'].transform(
        lambda x: x.shift(1).expanding().mean()
    ).fillna(0)
    
    # For Artifacts: We need to save the FINAL known stats for each jockey from the training set
    # so we can use it for inference (future data).
    final_jockey_stats = df.groupby('jockey_id')['is_win'].agg(['count', 'sum'])
    final_jockey_stats['rate'] = final_jockey_stats['sum'] / final_jockey_stats['count']
    jockey_win_rate_map = final_jockey_stats['rate'].to_dict()
    
    # Target Encoding (Trainer) - Expanding Window
    print("Calculating expanding window stats for Trainer Win Rate...")
    if 'trainer_id' not in df.columns:
        df['trainer_id'] = "unknown"
        
    df['trainer_win_rate'] = df.groupby('trainer_id')['is_win'].transform(
        lambda x: x.shift(1).expanding().mean()
    ).fillna(0)
    
    final_trainer_stats = df.groupby('trainer_id')['is_win'].agg(['count', 'sum'])
    final_trainer_stats['rate'] = final_trainer_stats['sum'] / final_trainer_stats['count']
    trainer_win_rate_map = final_trainer_stats['rate'].to_dict()

    # Target Encoding (Pedigree: Sire & DamSire)
    # Check if columns exist (merged from horse_profiles)
    sire_win_rate_map = {}
    damsire_win_rate_map = {}
    
    for col, name in [('sire_id', 'Sire'), ('damsire_id', 'DamSire')]:
        if col in df.columns:
            print(f"Calculating expanding window stats for {name} Win Rate...")
            # Fill missing IDs
            df[col] = df[col].astype(str).replace('nan', 'unknown').fillna('unknown')
            
            df[f'{col.replace("_id", "")}_win_rate'] = df.groupby(col)['is_win'].transform(
                lambda x: x.shift(1).expanding().mean()
            ).fillna(0)
            
            # Artifacts
            stats = df.groupby(col)['is_win'].agg(['count', 'sum'])
            stats['rate'] = stats['sum'] / stats['count']
            if col == 'sire_id':
                sire_win_rate_map = stats['rate'].to_dict()
            else:
                damsire_win_rate_map = stats['rate'].to_dict()
        else:
            print(f"Warning: {col} not found in data. Filling with 0.")
            df[f'{col.replace("_id", "")}_win_rate'] = 0.0

    # Feature: Weight Diff (Clean)
    # 484(+2) -> +2 extracted by scraper as 'weight_diff'. Ensure numeric.
    if 'weight_diff' not in df.columns:
        df['weight_diff'] = 0
    
    df['weight_diff'] = pd.to_numeric(df['weight_diff'], errors='coerce').fillna(0)

    # Artifacts storage
    from sklearn.preprocessing import LabelEncoder
    artifacts = {
        'jockey_win_rate': jockey_win_rate_map,
        'trainer_win_rate': trainer_win_rate_map,
        'sire_win_rate': sire_win_rate_map,       # New
        'damsire_win_rate': damsire_win_rate_map, # New
        'course_stats': None # Placeholder
    }
    
    # Save Course Stats for Speed Index (computed earlier) to artifacts
    if 'course_type' in df.columns and 'distance' in df.columns:
         valid_times = df[df['time_sec'] > 0]
         course_stats = valid_times.groupby(['course_type', 'distance'])['time_sec'].agg(['mean', 'std']).reset_index()
         # Convert to dict for easier serialization or keep as DF
         # Let's keep as DF but standardized columns
         course_stats.columns = ['course_type', 'distance', 'course_mean', 'course_std']
         artifacts['course_stats'] = course_stats.to_dict('records') # List of dicts

    # Encode IDs (Update CATEGORY_COLS later in settings, but handle here if added)
    for col in settings.CATEGORY_COLS:
        if col in df.columns:
            # Add string conversion for safety
            df[col] = df[col].astype(str).fillna("unknown")
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            artifacts[col] = le

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
    
    # Feature: Speed Index
    # Use Artifacts if available (preferred for consistency)
    if 'course_stats' in artifacts and artifacts['course_stats'] is not None:
        stats_data = artifacts['course_stats']
        # Convert back to DF
        stats_df = pd.DataFrame(stats_data)
        
        # Merge
        if 'course_type' in df.columns and 'distance' in df.columns:
            df = df.merge(stats_df, on=['course_type', 'distance'], how='left')
            df['speed_index'] = (df['course_mean'] - df['time_sec']) / df['course_std'].replace(0, 1)
            df['speed_index'] = df['speed_index'].fillna(0)
        else:
            df['speed_index'] = 0
    else:
        # Fallback: Calc on the fly (batch mode)
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
    
    # Ensure rank is numeric for lag calculation
    if 'rank' in df.columns:
        df['rank'] = pd.to_numeric(df['rank'], errors='coerce')
        
    df['lag1_rank'] = df.groupby('horse_id')['rank'].shift(1).fillna(99).astype(int)
    df['lag1_speed_index'] = df.groupby('horse_id')['speed_index'].shift(1).fillna(0)
    df['interval'] = (df['date'] - df.groupby('horse_id')['date'].shift(1)).dt.days.fillna(365)

    # Encoding using Artifacts
    # Added Pedigree Features
    encoding_cols = [
        ('jockey_win_rate', 'jockey_id'),
        ('trainer_win_rate', 'trainer_id'),
        ('sire_win_rate', 'sire_id'),
        ('damsire_win_rate', 'damsire_id')
    ]
    
    for col, enc_map in encoding_cols:
        if col in artifacts:
            map_dict = artifacts[col]
            # Map with type safety fallback
            def get_rate(key, m=map_dict):
                if key in m: return m[key]
                if str(key) in m: return m[str(key)]
                return 0.0
            id_col = enc_map
            if id_col in df.columns:
                # Ensure ID is processed (e.g. unknown) handled by map safety or pre-fill
                df[col] = df[id_col].astype(str).apply(get_rate)
            else:
                # Fallback if ID column missing (e.g. inference data lacks profile)
                df[col] = 0.0
        else:
            df[col] = 0.0
            
    # Weight Diff
    if 'weight_diff' in df.columns:
        df['weight_diff'] = pd.to_numeric(df['weight_diff'], errors='coerce').fillna(0)
    else:
        df['weight_diff'] = 0

    # Label Encoders
    for col in settings.CATEGORY_COLS:
        if col in df.columns:
            # Keys in encoders.pkl are bare column names (e.g. 'horse_id')
            if col in artifacts:
                le = artifacts[col]
                valid_classes = set(le.classes_)
                # Handle unknown
                df[col] = df[col].astype(str).map(lambda x: x if x in valid_classes else "unknown")
                if "unknown" not in valid_classes:
                    # Fallback to 0 index class if "unknown" not explicitly trained
                    fallback = list(valid_classes)[0]
                    df[col] = df[col].map(lambda x: x if x in valid_classes else fallback)
                
                df[col] = le.transform(df[col]).astype(int)

    # Rank Class (for evaluation if rank exists)
    if 'rank' in df.columns:
        df['rank'] = pd.to_numeric(df['rank'], errors='coerce')
        conditions = [(df['rank'] == 1), (df['rank'] <= 3), (df['rank'] <= 5)]
        choices = [0, 1, 2]
        df['rank_class'] = np.select(conditions, choices, default=3)
    
    df = df.fillna(0)
    return df

def split_data(df, valid_ratio=0.15):
    """
    Split into Train/Valid (Time Series Split).
    Test set is usually separate in this workflow (e.g. 2025).
    """
    # Sort by date
    df = df.sort_values('date').reset_index(drop=True)
    
    n = len(df)
    train_idx = int(n * (1 - valid_ratio))
    
    train = df.iloc[:train_idx]
    valid = df.iloc[train_idx:]
    
    # Check simple date boundaries
    if not train.empty and not valid.empty:
        print(f"Train: {train['date'].min()} -> {train['date'].max()} ({len(train)} rows)")
        print(f"Valid: {valid['date'].min()} -> {valid['date'].max()} ({len(valid)} rows)")
    
    return train, valid, None # No test set here
