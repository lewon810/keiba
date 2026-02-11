import joblib
import pandas as pd
import os
import sys

# Add project root to path to import train.settings if needed
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from train import settings
except ImportError:
    class settings:
        MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'train', 'data', 'model')
        MODEL_PATH = os.path.join(MODEL_DIR, 'model_lgb.pkl')

def predict(race_data, return_df=False, power=None):
    """
    Takes race data (list of dicts) and returns predictions using the trained model.
    If return_df is True, returns the pandas DataFrame with scores.
    power: exponent for score calculation (P^power * Odds), defaults to settings.POWER_EXPONENT
    """
    if not race_data:
        return "No data to predict."

    # Check if model exists
    encoder_path = os.path.join(settings.MODEL_DIR, 'encoders.pkl')
    if not os.path.exists(settings.MODEL_PATH) or not os.path.exists(encoder_path):
        return "Error: Model or encoders not found. Please train the model first."

    try:
        # Load Artifacts
        model = joblib.load(settings.MODEL_PATH)
        artifacts = joblib.load(encoder_path)

        # DataFrame
        df = pd.DataFrame(race_data)

        # --- Feature Engineering for Inference ---

        # 1. Load History (Lag Features)
        try:
            from .history_loader import loader
        except ImportError:
            from history_loader import loader

        try:
            loader.load() # Load CSVs once

            # Enrich race_data with history
            for i, row in df.iterrows():
                # Current Race Date provided by metadata?
                # Scraper puts "date" in input race_data! string "2024年..."
                current_date = row.get('date', None)

                last_stats = loader.get_last_race(row['horse_id'], current_date_str=current_date)

                if last_stats:
                    df.at[i, 'lag1_rank'] = last_stats['lag1_rank']
                    df.at[i, 'lag1_speed_index'] = last_stats['lag1_speed_index']
                    df.at[i, 'interval'] = last_stats['interval']
                else:
                    df.at[i, 'lag1_rank'] = 99
                    df.at[i, 'lag1_speed_index'] = 0
                    df.at[i, 'interval'] = 365
        except Exception as e:
            print(f"History load failed: {e}")
            df['lag1_rank'] = 99
            df['lag1_speed_index'] = 0
            df['interval'] = 365

        # 2. Jockey Win Rate
        jockey_map = artifacts.get('jockey_win_rate', {})
        def get_rate(jid):
            # Try exact match, then string match, then 0
            if jid in jockey_map: return jockey_map[jid]
            try:
                if int(jid) in jockey_map: return jockey_map[int(jid)]
            except: pass
            try:
                if str(jid) in jockey_map: return jockey_map[str(jid)]
            except: pass
            return 0.0

        df['jockey_win_rate'] = df['jockey_id'].apply(get_rate)

        # 2b. Trainer Win Rate
        trainer_map = artifacts.get('trainer_win_rate', {})
        def get_trainer_rate(tid):
            if tid in trainer_map: return trainer_map[tid]
            try:
                if int(tid) in trainer_map: return trainer_map[int(tid)]
            except: pass
            try:
                if str(tid) in trainer_map: return trainer_map[str(tid)]
            except: pass
            return 0.0
            
        df['trainer_win_rate'] = df['trainer_id'].apply(get_trainer_rate)

        # 3. Categorical Encoding (Label Encoder)
        cat_cols = ['horse_id', 'jockey_id', 'trainer_id', 'course_type', 'weather', 'condition', 'sire_id', 'damsire_id', 'running_style']
        
        # 2c. Sire/DamSire Win Rate
        for col in ['sire_win_rate', 'damsire_win_rate']:
            base_col = col.replace('_win_rate', '_id') # sire_id
            map_data = artifacts.get(col, {})
            def get_pedigree_rate(pid):
                if pid in map_data: return map_data[pid]
                if str(pid) in map_data: return map_data[str(pid)]
                return 0.0
            # Ensure base col exists first (handled in loop below? No, must exist for apply)
            if base_col not in df.columns: df[base_col] = 'unknown'
            df[col] = df[base_col].apply(get_pedigree_rate)

        # 2d. Aptitude Features (Turf/Dirt, Distance)
        # Turf/Dirt
        apt_type_map = artifacts.get('aptitude_type', {})
        def get_type_aptitude(row):
            hid = str(row['horse_id'])
            ctype = row.get('course_type', 'unknown')
            if hid in apt_type_map and ctype in apt_type_map[hid]:
                return apt_type_map[hid][ctype]
            return 0.0
        df['course_type_win_rate'] = df.apply(get_type_aptitude, axis=1)

        # Distance
        apt_dist_map = artifacts.get('aptitude_dist', {})
        def get_dist_cat(d):
            try:
                d = int(d)
                if d < 1400: return 'sprint'
                if d < 1900: return 'mile'
                if d < 2500: return 'intermediate'
                return 'long'
            except:
                return 'unknown'
        
        # Create temp dist_cat if needed
        df['dist_cat_temp'] = df['distance'].apply(get_dist_cat)
        
        def get_dist_aptitude(row):
            hid = str(row['horse_id'])
            cat = row.get('dist_cat_temp', 'unknown')
            if hid in apt_dist_map and cat in apt_dist_map[hid]:
                return apt_dist_map[hid][cat]
            return 0.0
        df['dist_cat_win_rate'] = df.apply(get_dist_aptitude, axis=1)


        for col in cat_cols:
            # Handle Pedigree/Style missing in input
            if col not in df.columns:
                 df[col] = "unknown"
                 
            # Keys in encoders.pkl are bare column names (e.g. 'horse_id')
            if col in artifacts:
                le = artifacts[col]
                valid_classes = set(le.classes_)
                # Handle unknown
                df[col] = df[col].astype(str).map(lambda x: x if x in valid_classes else "unknown")
                # If "unknown" itself is not in classes, map to index 0 safety
                if "unknown" not in valid_classes:
                     df[col] = df[col].map(lambda x: x if x in valid_classes else list(valid_classes)[0])

                df[col] = le.transform(df[col]).astype(int)
            else:
                 # If encoder missing, fill 0
                 df[col] = 0

        # ... (Numeric cleanup skipped in this diff, assuming follow-up or inclusion)
        # 4. Numeric cleanup
        df['waku'] = pd.to_numeric(df['waku'], errors='coerce').fillna(0)
        df['umaban'] = pd.to_numeric(df['umaban'], errors='coerce').fillna(0)
        df['distance'] = pd.to_numeric(df['distance'], errors='coerce').fillna(0)
        
        # Missing columns handling
        if 'weight_diff' not in df.columns:
            df['weight_diff'] = 0
        df['weight_diff'] = pd.to_numeric(df['weight_diff'], errors='coerce').fillna(0)

        # 5. Predict
        features = [
            'jockey_win_rate', 'trainer_win_rate', 'horse_id', 'jockey_id', 'trainer_id',
            'waku', 'umaban', 'course_type', 'distance', 'weather', 'condition',
            'lag1_rank', 'lag1_speed_index', 'interval', 'weight_diff',
            'sire_id', 'damsire_id', 'running_style',
            'sire_win_rate', 'damsire_win_rate',
            'course_type_win_rate', 'dist_cat_win_rate'
        ]

        # LambdaRank returns 1D score array (N,) - higher is better
        pred_scores = model.predict(df[features])
        
        # Use scores directly as win probability proxy
        # Normalize to 0-1 range for better interpretability
        if len(pred_scores) > 0:
            min_score = pred_scores.min()
            max_score = pred_scores.max()
            if max_score > min_score:
                df['win_prob'] = (pred_scores - min_score) / (max_score - min_score)
            else:
                df['win_prob'] = 0.5  # All same score
        else:
            df['win_prob'] = 0.0

        # Clean Odds for calculation
        def parse_odds(o):
            try:
                return float(o)
            except:
                return 0.0
        df['odds_val'] = df['odds'].apply(parse_odds)

        # Calculate Score: (Win Prob)^4 * Odds
        # If odds are missing (0.0), score becomes 0.
        # Fallback to win_prob if odds are missing? 
        # Strategy implies odds are crucial. If odds 0 (e.g. new race w/o odds), 
        # this strategy fails. Assuming odds exist or fallback to prob.
        
        # Hybrid Score: Use Expectation if odds exist, else raw prob
        use_power = power if power is not None else settings.POWER_EXPONENT
        df['score'] = df.apply(
            lambda x: (x['win_prob'] ** use_power) * x['odds_val'] if x['odds_val'] > 0 else x['win_prob'], 
            axis=1
        )

        # Normalize scores to 0-1 range for better readability
        # Note: Expectation scores can be widely distributed
        # User requested RAW score: P^4 * Odds
        # normalization removed
        pass

        # Rank by Score (Descending)
        df = df.sort_values('score', ascending=False)
        
        if return_df:
            return df

        # 6. Format Output
        # Get context from original race_data to avoid showing encoded integers
        context_weather = race_data[0].get('weather', 'Unknown')
        context_distance = race_data[0].get('distance', 'Unknown')

        result_lines = [f"Prediction Ranking (Score = Prob^{use_power} * Odds):"]
        result_lines.append(f"Context: {context_weather} / {context_distance}m")
        result_lines.append("-" * 40)

        for i, (_, row) in enumerate(df.iterrows()):
            symbol = "  "
            if i == 0: symbol = "◎ "
            elif i == 1: symbol = "○ "
            elif i == 2: symbol = "▲ "
            elif i == 3: symbol = "△ "

            # Show odds if available, else ---
            odds_str = str(row.get('odds', '---.-'))
            
            # Show Probability as well for transparency
            prob_pct = row['win_prob'] * 100
            
            line = f"{symbol} {i+1}. {row['name']} (Odds: {odds_str}, Win%: {prob_pct:.1f}%, Score: {row['score']:.4f})"
            result_lines.append(line)

        return "\n".join(result_lines)

    except Exception as e:
        import traceback
        return f"Prediction Error: {e}\n{traceback.format_exc()}"