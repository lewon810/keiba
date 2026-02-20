import pandas as pd
import numpy as np
import os
from . import settings

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
    
    # Feature: Last 3F (上がり3ハロン)
    # Parse last_3f to numeric seconds
    if 'last_3f' in df.columns:
        df['last_3f_time'] = pd.to_numeric(df['last_3f'], errors='coerce').fillna(0)
        
        # Calculate last_3f rank within each race
        df['last_3f_rank'] = df.groupby('race_id')['last_3f_time'].rank(method='min', ascending=True).fillna(99)
        
        # Calculate last_3f deviation score (偏差値: mean=50, std=10)
        # Group by race to get relative performance
        race_3f_stats = df.groupby('race_id')['last_3f_time'].agg(['mean', 'std']).reset_index()
        race_3f_stats.columns = ['race_id', 'race_3f_mean', 'race_3f_std']
        df = df.merge(race_3f_stats, on='race_id', how='left')
        
        # Deviation score: 50 - (value - mean) / std * 10
        # Lower last_3f_time is better (faster), so we invert
        df['last_3f_deviation'] = 50 - ((df['last_3f_time'] - df['race_3f_mean']) / df['race_3f_std'].replace(0, 1)) * 10
        df['last_3f_deviation'] = df['last_3f_deviation'].fillna(50)  # Default to average
        
        # Drop temporary columns
        df = df.drop(columns=['race_3f_mean', 'race_3f_std'], errors='ignore')
    else:
        df['last_3f_time'] = 0
        df['last_3f_rank'] = 99
        df['last_3f_deviation'] = 50
    
    # Feature: Pace (ペース情報)
    # Count front-runners (逃げ・先行) in each race based on passing position
    if 'passing' in df.columns:
        # Extract first corner position from passing (e.g., "4-4" -> 4)
        def get_first_position(passing):
            if not passing or not isinstance(passing, str) or '-' not in passing:
                return 99
            try:
                pos_list = [int(p) for p in passing.split('-') if p.isdigit()]
                return pos_list[0] if pos_list else 99
            except:
                return 99
        
        df['first_position'] = df['passing'].apply(get_first_position)
        
        # Count front runners (position <= 2) per race
        df['is_front_runner'] = (df['first_position'] <= 2).astype(int)
        race_pace = df.groupby('race_id').agg({
            'is_front_runner': 'sum',
            'horse_id': 'count'  # Total horses in race
        }).reset_index()
        race_pace.columns = ['race_id', 'front_runner_count', 'race_size']
        
        df = df.merge(race_pace, on='race_id', how='left')
        
        # Pace ratio: front_runner_count / race_size
        df['pace_ratio'] = df['front_runner_count'] / df['race_size'].replace(0, 1)
        df['pace_ratio'] = df['pace_ratio'].fillna(0)
        
        # Drop temporary columns
        df = df.drop(columns=['first_position', 'is_front_runner', 'race_size'], errors='ignore')
    else:
        df['front_runner_count'] = 0
        df['pace_ratio'] = 0
    
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
    
    # Lag 1: Previous Last 3F Time (前走の上がり3F)
    # This is VALID - it's previous race data, not current race (no leakage)
    df['lag1_last_3f'] = df.groupby('horse_id')['last_3f_time'].shift(1).fillna(0)
    
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

    # Feature: Running Style (脚質) [Audit Recommendation]
    # Based on 'passing' column (e.g. 1-1-2-2)
    def extract_running_style(passing):
        if not passing or not isinstance(passing, str) or '-' not in passing:
            return "unknown"
        try:
            # Get first corner position
            pos_list = [int(p) for p in passing.split('-') if p.isdigit()]
            if not pos_list: return "unknown"
            
            first_pos = pos_list[0]
            # Simple heuristic:
            if first_pos <= 2: return "front" # 逃げ・先行
            if first_pos <= 7: return "middle" # 先行・差し
            return "back" # 差し・追込
        except:
            return "unknown"
            
    if 'passing' in df.columns:
        df['running_style'] = df['passing'].apply(extract_running_style)
    else:
        df['running_style'] = "unknown"

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
        def get_dist_cat(d):
            try:
                d = int(d)
                if d < 1400: return 'sprint'
                if d < 1900: return 'mile'
                if d < 2500: return 'intermediate'
                return 'long'
            except:
                return 'unknown'
                
        df['dist_cat'] = df['distance'].apply(get_dist_cat)
        
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
    # 日付のパース: year, month, day カラムから datetime を構築
    if 'year' in df.columns and 'month' in df.columns and 'day' in df.columns:
        df['date'] = pd.to_datetime(df[['year', 'month', 'day']], errors='coerce')
    elif 'date' in df.columns:
        # レガシーフォールバック: 旧フォーマット対応
        if df['date'].dtype == 'int64' or df['date'].dtype == 'int32':
            def extract_date_from_race_id(rid):
                try:
                    rid_str = str(rid)
                    if len(rid_str) >= 12:
                        year = rid_str[0:4]
                        month = rid_str[6:8]
                        day = rid_str[8:10]
                        return pd.to_datetime(f"{year}-{month}-{day}", errors='coerce')
                    return pd.NaT
                except:
                    return pd.NaT
            df['date'] = df['race_id'].apply(extract_date_from_race_id)
        else:
            df['date'] = pd.to_datetime(df['date'], format='%Y年%m月%d日', errors='coerce')
    else:
        df['date'] = pd.NaT
    
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
    
    # Feature: Last 3F (上がり3ハロン) - Same logic as preprocess
    if 'last_3f' in df.columns:
        df['last_3f_time'] = pd.to_numeric(df['last_3f'], errors='coerce').fillna(0)
        
        # Calculate last_3f rank within each race
        df['last_3f_rank'] = df.groupby('race_id')['last_3f_time'].rank(method='min', ascending=True).fillna(99)
        
        # Calculate last_3f deviation score
        race_3f_stats = df.groupby('race_id')['last_3f_time'].agg(['mean', 'std']).reset_index()
        race_3f_stats.columns = ['race_id', 'race_3f_mean', 'race_3f_std']
        df = df.merge(race_3f_stats, on='race_id', how='left')
        
        df['last_3f_deviation'] = 50 - ((df['last_3f_time'] - df['race_3f_mean']) / df['race_3f_std'].replace(0, 1)) * 10
        df['last_3f_deviation'] = df['last_3f_deviation'].fillna(50)
        
        df = df.drop(columns=['race_3f_mean', 'race_3f_std'], errors='ignore')
    else:
        df['last_3f_time'] = 0
        df['last_3f_rank'] = 99
        df['last_3f_deviation'] = 50
    
    # Feature: Pace (ペース情報) - Same logic as preprocess
    if 'passing' in df.columns:
        def get_first_position(passing):
            if not passing or not isinstance(passing, str) or '-' not in passing:
                return 99
            try:
                pos_list = [int(p) for p in passing.split('-') if p.isdigit()]
                return pos_list[0] if pos_list else 99
            except:
                return 99
        
        df['first_position'] = df['passing'].apply(get_first_position)
        df['is_front_runner'] = (df['first_position'] <= 2).astype(int)
        
        race_pace = df.groupby('race_id').agg({
            'is_front_runner': 'sum',
            'horse_id': 'count'
        }).reset_index()
        race_pace.columns = ['race_id', 'front_runner_count', 'race_size']
        
        df = df.merge(race_pace, on='race_id', how='left')
        df['pace_ratio'] = df['front_runner_count'] / df['race_size'].replace(0, 1)
        df['pace_ratio'] = df['pace_ratio'].fillna(0)
        
        df = df.drop(columns=['first_position', 'is_front_runner', 'race_size'], errors='ignore')
    else:
        df['front_runner_count'] = 0
        df['pace_ratio'] = 0
    
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

    # Feature: Running Style (Validation Only - Leakage for Inference if using current passing)
    # If passing exists (results data), calculate it. Else unknown.
    def extract_running_style(passing):
        if not passing or not isinstance(passing, str) or '-' not in passing:
            return "unknown"
        try:
            # Get first corner position
            pos_list = [int(p) for p in passing.split('-') if p.isdigit()]
            if not pos_list: return "unknown"
            
            first_pos = pos_list[0]
            # Simple heuristic:
            if first_pos <= 2: return "front" # 逃げ・先行
            if first_pos <= 7: return "middle" # 先行・差し
            return "back" # 差し・追込
        except:
            return "unknown"
            
    if 'passing' in df.columns:
        df['running_style'] = df['passing'].apply(extract_running_style)
    else:
        df['running_style'] = "unknown"

    # Lag Features (Past Performance) - Self-contained sort
    df = df.sort_values(['horse_id', 'date'])
    
    # Ensure rank is numeric for lag calculation, create if missing (inference)
    if 'rank' in df.columns:
        df['rank'] = pd.to_numeric(df['rank'], errors='coerce')
    else:
        df['rank'] = np.nan
        
    df['lag1_rank'] = df.groupby('horse_id')['rank'].shift(1).fillna(99).astype(int)
    df['lag1_speed_index'] = df.groupby('horse_id')['speed_index'].shift(1).fillna(0)
    
    # Lag 1: Previous Last 3F Time (前走の上がり3F)
    df['lag1_last_3f'] = df.groupby('horse_id')['last_3f_time'].shift(1).fillna(0)
    
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
            
    # Aptitude Features Application (Inference)
    # Apply using aptitude maps
    if 'aptitude_type' in artifacts:
        type_map = artifacts['aptitude_type']
        def get_type_aptitude(row):
            hid = str(row['horse_id'])
            ctype = row.get('course_type', 'unknown')
            if hid in type_map and ctype in type_map[hid]:
                return type_map[hid][ctype]
            return 0.0
        df['course_type_win_rate'] = df.apply(get_type_aptitude, axis=1)
    else:
        df['course_type_win_rate'] = 0.0

    if 'aptitude_dist' in artifacts:
        dist_map = artifacts['aptitude_dist']
        def get_dist_cat(d):
            try:
                d = int(d)
                if d < 1400: return 'sprint'
                if d < 1900: return 'mile'
                if d < 2500: return 'intermediate'
                return 'long'
            except:
                return 'unknown'
        
        # Ensure dist_cat exists
        df['dist_cat'] = df['distance'].apply(get_dist_cat)
        
        def get_dist_aptitude(row):
            hid = str(row['horse_id'])
            cat = row.get('dist_cat', 'unknown')
            if hid in dist_map and cat in dist_map[hid]:
                return dist_map[hid][cat]
            return 0.0
        df['dist_cat_win_rate'] = df.apply(get_dist_aptitude, axis=1)
    else:
        df['dist_cat_win_rate'] = 0.0
            
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
