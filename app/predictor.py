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

def predict(race_data):
    """
    Takes race data (list of dicts) and returns predictions using the trained model.
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

        # 3. Categorical Encoding (Label Encoder)
        cat_cols = ['horse_id', 'jockey_id', 'course_type', 'weather', 'condition']
        # 'trainer_id' not in minimal scraper yet

        for col in cat_cols:
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
                 # If encoder missing, fill 0 (though this shouldn't happen if trained correctly)
                 print(f"Warning: Encoder for {col} not found.")
                 df[col] = 0

        # 4. Numeric cleanup
        df['waku'] = pd.to_numeric(df['waku'], errors='coerce').fillna(0)
        df['umaban'] = pd.to_numeric(df['umaban'], errors='coerce').fillna(0)
        df['distance'] = pd.to_numeric(df['distance'], errors='coerce').fillna(0)

        # 5. Predict
        features = [
            'jockey_win_rate', 'horse_id', 'jockey_id', 'waku', 'umaban',
            'course_type', 'distance', 'weather', 'condition',
            'lag1_rank', 'lag1_speed_index', 'interval'
        ]

        # LightGBM Multiclass returns (N, 4) probability matrix
        pred_probs = model.predict(df[features])
        
        # Extract Win Probability (Class 0)
        if pred_probs.ndim > 1:
            df['win_prob'] = pred_probs[:, 0]
        else:
            df['win_prob'] = pred_probs

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
        df['score'] = df.apply(
            lambda x: (x['win_prob'] ** 4) * x['odds_val'] if x['odds_val'] > 0 else x['win_prob'], 
            axis=1
        )

        # Normalize scores to 0-1 range for better readability
        # Note: Expectation scores can be widely distributed
        min_score = df['score'].min()
        max_score = df['score'].max()
        if max_score > min_score:
            df['score'] = (df['score'] - min_score) / (max_score - min_score)
        else:
            df['score'] = 0.5  # Handle case where all scores are the same

        # Rank by Score (Descending)
        df = df.sort_values('score', ascending=False)

        # 6. Format Output
        # Get context from original race_data to avoid showing encoded integers
        context_weather = race_data[0].get('weather', 'Unknown')
        context_distance = race_data[0].get('distance', 'Unknown')

        result_lines = ["Prediction Ranking (Score = Prob^4 * Odds):"]
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