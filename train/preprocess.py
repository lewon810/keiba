import pandas as pd
import numpy as np
import os
from . import settings
from . import features as feat

def load_data(start_year=None, end_year=None, start_month=None, end_month=None):
    """Loads all result CSVs from raw data directory, optionally filtering by year and month."""
    # Ensure we only load results_*.csv files, excluding things like horse_profiles.csv
    files = [f for f in os.listdir(settings.RAW_DATA_DIR) if f.startswith('results_') and f.endswith('.csv')]
    dfs = []
    
    # results_YYYY.csv という形式のファイル名から年を抽出してフィルタリング
    target_files = []
    if start_year and end_year:
        target_years = range(start_year, end_year + 1)
        for f in files:
            # "results_YYYY.csv" 形式のみ対象（アンダースコアがある場合はスキップ）
            try:
                parts = f.replace('results_', '').replace('.csv', '')
                if '_' in parts:
                    # results_YYYY_ZZZZ.csv のような形式は対象外
                    continue
                y = int(parts)
                if y in target_years:
                    target_files.append(f)
            except (ValueError, TypeError):
                pass
    else:
        target_files = files

    print(f"Loading data from: {target_files}")

    for f in target_files:
        path = os.path.join(settings.RAW_DATA_DIR, f)
        try:
            # Dtype optimized to prevent warnings - race_id must be str for month extraction
            df = pd.read_csv(path, dtype={'race_id': str, 'horse_id': str, 'jockey_id': str, 'trainer_id': str})
            # Drop invalid dates if any
            # もし日付列が存在する場合、数値型への変換や修正が必要な場合がありますが、
            # 基本的には後続の処理で上書きまたはパースされます
            pass
            dfs.append(df)
        except Exception as e:
            print(f"Skipping {f}: {e}")
            
    if not dfs:
        # Fallback or raise
        print("No matching data found.")
        return pd.DataFrame()
        
    df = pd.concat(dfs, ignore_index=True)
    
    # Month filtering if specified
    if start_month is not None or end_month is not None:
        # CSVに存在する month カラムを直接使用
        # 注意: race_id[4:6] は競馬場コードであり月ではない
        if 'month' in df.columns:
            df['_temp_month'] = pd.to_numeric(df['month'], errors='coerce')
        else:
            # フォールバック: month カラムがない場合は警告
            print("Warning: 'month' column not found in data. Cannot filter by month.")
            df['_temp_month'] = None
        
        # Filter by month range
        if start_month is not None and end_month is not None:
            initial_len = len(df)
            df = df[(df['_temp_month'] >= start_month) & (df['_temp_month'] <= end_month)]
            print(f"Filtered by month {start_month}-{end_month}: {initial_len} -> {len(df)} rows")
        elif start_month is not None:
            initial_len = len(df)
            df = df[df['_temp_month'] >= start_month]
            print(f"Filtered by month >= {start_month}: {initial_len} -> {len(df)} rows")
        elif end_month is not None:
            initial_len = len(df)
            df = df[df['_temp_month'] <= end_month]
            print(f"Filtered by month <= {end_month}: {initial_len} -> {len(df)} rows")
        
        df = df.drop(columns=['_temp_month'])
    
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
            # IDを文字列型に変換
            if 'horse_id' in profiles.columns:
                profiles['horse_id'] = profiles['horse_id'].astype(str)
                # 必要な列のみマージ
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
    
    # 日付のパース
    # year, month, day カラムから datetime を構築
    df['date'] = pd.to_datetime(df[['year', 'month', 'day']].rename(
        columns={'year': 'year', 'month': 'month', 'day': 'day'}), errors='coerce')
    
    # Feature: Time (seconds) — 共通関数を使用
    df['time_sec'] = df['time'].apply(feat.parse_time)
    
    # Feature: Last 3F (上がり3ハロン) — 共通関数を使用
    df = feat.compute_last_3f_features(df)
    
    # Feature: Popularity (人気順位) — レース前に判明する情報
    if 'popularity' in df.columns:
        df['popularity'] = pd.to_numeric(df['popularity'], errors='coerce').fillna(99)
    else:
        df['popularity'] = 99
    
    # Feature: Number of Runners (出走頭数)
    df['num_runners'] = df.groupby('race_id')['horse_id'].transform('count')
    
    # Feature: Popularity Ratio (相対人気) — 人気の絶対値ではなく出走頭数に対する相対値
    # 例: 18頭立て1番人気 → 1/18 ≈ 0.056、18頭立て18番人気 → 1.0
    # 値が低いほど人気が高い（人気1位で最小値）
    df['popularity_ratio'] = df['popularity'] / df['num_runners'].replace(0, 1)
    
    # Feature: Horse Age (馬齢) — horse_id の先頭4桁が生年
    if 'horse_id' in df.columns:
        df['horse_birth_year'] = df['horse_id'].astype(str).str[:4]
        df['horse_birth_year'] = pd.to_numeric(df['horse_birth_year'], errors='coerce')
        df['horse_age'] = df['year'] - df['horse_birth_year']
        df['horse_age'] = df['horse_age'].clip(lower=2, upper=10).fillna(3)
        df = df.drop(columns=['horse_birth_year'], errors='ignore')
    else:
        df['horse_age'] = 3
    
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

    # Feature: Running Style (脚質) — 前走の通過順から算出（リーケージ回避）
    if 'passing' in df.columns:
        df['first_position'] = df['passing'].apply(feat.get_first_position)
    else:
        df['first_position'] = 99
    
    # Feature: Lag Features (Past Performance)
    # Sort by Horse and Date
    df = df.sort_values(['horse_id', 'date'])
    
    # Lag 1: Previous Rank
    df['lag1_rank'] = df.groupby('horse_id')['rank'].shift(1).fillna(99)
    
    # Lag 2, 3: 2走前・3走前の着順
    df['lag2_rank'] = df.groupby('horse_id')['rank'].shift(2).fillna(99)
    df['lag3_rank'] = df.groupby('horse_id')['rank'].shift(3).fillna(99)
    
    # Average of last 3 ranks (直近3走の平均着順)
    df['avg_last3_rank'] = df[['lag1_rank', 'lag2_rank', 'lag3_rank']].mean(axis=1)
    
    # 正規化: 絶対着順(1-99)を 0.0〜1.0 スケールへ変換
    # clip(1, 10)で10着以降は区別せず、(val-1)/9.0で正規化
    # 1着→0.0（最良）, 10着以上→1.0（最悪）
    df['lag1_rank_norm'] = (df['lag1_rank'].clip(1, 10) - 1) / 9.0
    df['lag2_rank_norm'] = (df['lag2_rank'].clip(1, 10) - 1) / 9.0
    df['lag3_rank_norm'] = (df['lag3_rank'].clip(1, 10) - 1) / 9.0
    df['avg_last3_rank_norm'] = (df['avg_last3_rank'].clip(1, 10) - 1) / 9.0
    
    # Lag 1: Previous Speed Index
    df['lag1_speed_index'] = df.groupby('horse_id')['speed_index'].shift(1).fillna(0)
    
    # Lag 1: Previous Last 3F Time (前走の上がり3F)
    df['lag1_last_3f'] = df.groupby('horse_id')['last_3f_time'].shift(1).fillna(0)
    
    # Lag 1: Interval (Days since last race)
    df['interval'] = (df['date'] - df.groupby('horse_id')['date'].shift(1)).dt.days.fillna(365)
    
    # Lag 1: Running Style — 前走の脚質（リーケージ回避）
    df['lag1_first_position'] = df.groupby('horse_id')['first_position'].shift(1).fillna(99)
    df['running_style'] = df['lag1_first_position'].apply(feat.classify_running_style)
    df = df.drop(columns=['first_position', 'lag1_first_position'], errors='ignore')

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

    # Note: running_style は上記の Lag Features セクションで
    # 前走の通過順からリーケージなしで算出済み

    # Feature: Aptitude (Turf/Dirt, Distance) - Expanding Window
    # Must be done after sorting by date (already sorted)
    print("Calculating Aptitude Features (Turf/Dirt, Distance)...")
    
    # Turf/Dirt Win Rate
    # Group by horse_id and course_type
    if 'course_type' in df.columns:
        # Create separate columns for turf and dirt rates
        # We need to pivot or calculate conditionally
        # Easier: Expanding mean of is_win * (course_type == 'turf')? No.
        # Subset approach:
        # Calculate expanding mean WITHIN the subgroup of (horse, type)
        # Then merge back? Or transform?
        
        # We want 'horse_turf_win_rate' on the row of a Turf race to represent PAST Turf performance.
        # But for inference, we want the Last Known Turf Rate regardless of current race type? 
        # Usually checking "Turf Aptitude" for a Turf race is what matters.
        # If a horse runs in Dirt, its Turf Aptitude is static (previous val).
        
        # Strategy: Calculate expanding stats per (horse, type), then forward fill per horse?
        # Simpler for now: Just calculate expanding rate given the current race context.
        # If I am running in Turf, use my past Turf stats.
        
        # 1. Group by [horse, type], cal expanding mean
        df['course_type_win_rate'] = df.groupby(['horse_id', 'course_type'])['is_win'].transform(
            lambda x: x.shift(1).expanding().mean()
        ).fillna(0)
        
        # 2. Extract specific columns for artifacts/inspection if needed, but 'course_type_win_rate' 
        # is the effective feature for the model (interaction term handles the rest).
        # But user requested specific "Turf Aptitude", "Dirt Aptitude".
        # Let's pivot to explicit columns for all rows if possible, but that's expensive (expanding per type).
        # Let's stick to 'same_type_win_rate' (aptitude for THIS race type).
        
        # Wait, for artifacts we need to store the map: HorseID -> {Turf: 0.5, Dirt: 0.1}
        # Final stats per horse per type
        final_type_stats = df.groupby(['horse_id', 'course_type'])['is_win'].agg(['count', 'sum']).reset_index()
        final_type_stats['rate'] = final_type_stats['sum'] / final_type_stats['count']
        
        # Convert to nested dict: {horse_id: {turf: 0.5, dirt: 0.0}}
        aptitude_type_map = {}
        for _, row in final_type_stats.iterrows():
            hid = str(row['horse_id'])
            ctype = row['course_type']
            if hid not in aptitude_type_map: aptitude_type_map[hid] = {}
            aptitude_type_map[hid][ctype] = row['rate']
    else:
        aptitude_type_map = {}
        
    # Distance Category Win Rate
    # Sprint: <1400, Mile: 1400-1899, Intermediate: 1900-2400, Long: >2400
    if 'distance' in df.columns:
        df['dist_cat'] = df['distance'].apply(feat.get_dist_cat)
        
        # Expanding mean per (horse, dist_cat)
        df['dist_cat_win_rate'] = df.groupby(['horse_id', 'dist_cat'])['is_win'].transform(
            lambda x: x.shift(1).expanding().mean()
        ).fillna(0)
        
        # Artifacts
        final_dist_stats = df.groupby(['horse_id', 'dist_cat'])['is_win'].agg(['count', 'sum']).reset_index()
        final_dist_stats['rate'] = final_dist_stats['sum'] / final_dist_stats['count']
        
        aptitude_dist_map = {}
        for _, row in final_dist_stats.iterrows():
            hid = str(row['horse_id'])
            cat = row['dist_cat']
            if hid not in aptitude_dist_map: aptitude_dist_map[hid] = {}
            aptitude_dist_map[hid][cat] = row['rate']
    else:
        aptitude_dist_map = {}


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
        'sire_win_rate': sire_win_rate_map,
        'damsire_win_rate': damsire_win_rate_map,
        'aptitude_type': aptitude_type_map, # New
        'aptitude_dist': aptitude_dist_map, # New
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
            le.fit(df[col])
            artifacts[col] = le
            df = feat.apply_label_encoder(df, col, le)

    # Fill NaNs
    df = df.fillna(0)
    
    return df, artifacts

def transform(df, artifacts):
    """
    Apply preprocessing using existing artifacts (Encoders, Maps).
    Used for Inference and Evaluation on new data.
    """
    # 日付のパース: year, month, day カラムから datetime を構築
    if 'year' in df.columns and 'month' in df.columns and 'day' in df.columns:
        df['date'] = pd.to_datetime(df[['year', 'month', 'day']], errors='coerce')
    elif 'date' in df.columns:
        # レガシーフォールバック: 旧フォーマット対応
        if df['date'].dtype == 'int64' or df['date'].dtype == 'int32':
            df['date'] = df['race_id'].apply(feat.extract_date_from_race_id)
        else:
            df['date'] = pd.to_datetime(df['date'], format='%Y年%m月%d日', errors='coerce')
    else:
        df['date'] = pd.NaT
    
    # Feature: Time (seconds) — 共通関数を使用
    df['time_sec'] = df['time'].apply(feat.parse_time)
    
    # Feature: Last 3F (上がり3ハロン) — 共通関数を使用
    df = feat.compute_last_3f_features(df)
    
    # Feature: Popularity (人気順位) — レース前に判明する情報
    if 'popularity' in df.columns:
        df['popularity'] = pd.to_numeric(df['popularity'], errors='coerce').fillna(99)
    else:
        df['popularity'] = 99
    
    # Feature: Number of Runners (出走頭数)
    df['num_runners'] = df.groupby('race_id')['horse_id'].transform('count')
    
    # Feature: Popularity Ratio (相対人気) — 人気の絶対値ではなく出走頭数に対する相対値
    # 例: 18頭立て1番人気 → 1/18 ≈ 0.056、18頭立て18番人気 → 1.0
    # 値が低いほど人気が高い（人気1位で最小値）
    df['popularity_ratio'] = df['popularity'] / df['num_runners'].replace(0, 1)
    
    # Feature: Horse Age (馬齢) — horse_id の先頭4桁が生年
    if 'horse_id' in df.columns and 'year' in df.columns:
        df['horse_birth_year'] = df['horse_id'].astype(str).str[:4]
        df['horse_birth_year'] = pd.to_numeric(df['horse_birth_year'], errors='coerce')
        df['horse_age'] = pd.to_numeric(df['year'], errors='coerce') - df['horse_birth_year']
        df['horse_age'] = df['horse_age'].clip(lower=2, upper=10).fillna(3)
        df = df.drop(columns=['horse_birth_year'], errors='ignore')
    else:
        df['horse_age'] = 3
    
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

    # Feature: Running Style (脚質) — 前走の通過順から算出（リーケージ回避）
    if 'passing' in df.columns:
        df['first_position'] = df['passing'].apply(feat.get_first_position)
    else:
        df['first_position'] = 99

    # Lag Features (Past Performance) - Self-contained sort
    df = df.sort_values(['horse_id', 'date'])
    
    # Ensure rank is numeric for lag calculation, create if missing (inference)
    if 'rank' in df.columns:
        df['rank'] = pd.to_numeric(df['rank'], errors='coerce')
    else:
        df['rank'] = np.nan
        
    df['lag1_rank'] = df.groupby('horse_id')['rank'].shift(1).fillna(99).astype(int)
    df['lag2_rank'] = df.groupby('horse_id')['rank'].shift(2).fillna(99).astype(int)
    df['lag3_rank'] = df.groupby('horse_id')['rank'].shift(3).fillna(99).astype(int)
    df['avg_last3_rank'] = df[['lag1_rank', 'lag2_rank', 'lag3_rank']].mean(axis=1)
    
    # 正規化: 絶対着順(1-99)を 0.0〜1.0 スケールへ変換
    # 1着→0.0（最良）, 10着以上→1.0（最悪）
    df['lag1_rank_norm'] = (df['lag1_rank'].clip(1, 10) - 1) / 9.0
    df['lag2_rank_norm'] = (df['lag2_rank'].clip(1, 10) - 1) / 9.0
    df['lag3_rank_norm'] = (df['lag3_rank'].clip(1, 10) - 1) / 9.0
    df['avg_last3_rank_norm'] = (df['avg_last3_rank'].clip(1, 10) - 1) / 9.0
    
    df['lag1_speed_index'] = df.groupby('horse_id')['speed_index'].shift(1).fillna(0)
    
    # Lag 1: Previous Last 3F Time (前走の上がり3F)
    df['lag1_last_3f'] = df.groupby('horse_id')['last_3f_time'].shift(1).fillna(0)
    
    df['interval'] = (df['date'] - df.groupby('horse_id')['date'].shift(1)).dt.days.fillna(365)
    
    # Lag 1: Running Style — 前走の脚質（リーケージ回避）
    df['lag1_first_position'] = df.groupby('horse_id')['first_position'].shift(1).fillna(99)
    df['running_style'] = df['lag1_first_position'].apply(feat.classify_running_style)
    df = df.drop(columns=['first_position', 'lag1_first_position'], errors='ignore')

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
            id_col = enc_map
            if id_col in df.columns:
                df[col] = df[id_col].astype(str).apply(lambda k: feat.lookup_rate(k, map_dict))
            else:
                df[col] = 0.0
        else:
            df[col] = 0.0
            
    # Aptitude Features Application (Inference)
    # Apply using aptitude maps
    if 'aptitude_type' in artifacts:
        type_map = artifacts['aptitude_type']
        def _get_type_aptitude(row):
            hid = str(row['horse_id'])
            ctype = row.get('course_type', 'unknown')
            if hid in type_map and ctype in type_map[hid]:
                return type_map[hid][ctype]
            return 0.0
        df['course_type_win_rate'] = df.apply(_get_type_aptitude, axis=1)
    else:
        df['course_type_win_rate'] = 0.0

    if 'aptitude_dist' in artifacts:
        dist_map = artifacts['aptitude_dist']
        df['dist_cat'] = df['distance'].apply(feat.get_dist_cat)
        def _get_dist_aptitude(row):
            hid = str(row['horse_id'])
            cat = row.get('dist_cat', 'unknown')
            if hid in dist_map and cat in dist_map[hid]:
                return dist_map[hid][cat]
            return 0.0
        df['dist_cat_win_rate'] = df.apply(_get_dist_aptitude, axis=1)
    else:
        df['dist_cat_win_rate'] = 0.0
            
    # Weight Diff
    if 'weight_diff' in df.columns:
        df['weight_diff'] = pd.to_numeric(df['weight_diff'], errors='coerce').fillna(0)
    else:
        df['weight_diff'] = 0

    # Label Encoders — 共通関数を使用
    for col in settings.CATEGORY_COLS:
        if col in df.columns:
            if col in artifacts:
                df = feat.apply_label_encoder(df, col, artifacts[col])

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
