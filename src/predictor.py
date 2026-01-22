import joblib
import pandas as pd
import os
import sys

# Add project root to path to import learn.settings if needed
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from learn import settings
except ImportError:
    class settings:
        MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'learn', 'data', 'model')
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
        
        # 1. Fill default Lag Features (Since we don't scrape past data yet)
        if 'lag1_rank' not in df.columns:
            df['lag1_rank'] = 99 # Default
        if 'lag1_speed_index' not in df.columns:
            df['lag1_speed_index'] = 0 # Default
        if 'interval' not in df.columns:
            df['interval'] = 365 # Default
        
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
            le_key = f'le_{col}'
            if le_key in artifacts:
                le = artifacts[le_key]
                valid_classes = set(le.classes_)
                # Handle unknown
                df[col] = df[col].astype(str).map(lambda x: x if x in valid_classes else "unknown")
                # If "unknown" itself is not in classes, map to index 0 safety
                if "unknown" not in valid_classes:
                     df[col] = df[col].map(lambda x: x if x in valid_classes else list(valid_classes)[0])
                
                df[col] = le.transform(df[col])
        
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
        
        preds = model.predict(df[features]) 
        df['prob_win'] = preds[:, 0]
        
        # 6. Strategy: Score = P^4 * Odds
        def parse_odds(x):
            try:
               return float(x)
            except:
               return 1.0 
               
        df['odds_val'] = df['odds'].apply(parse_odds)
        
        # Power of 4 strategy
        df['score'] = (df['prob_win'] ** 4) * df['odds_val']
        
        # Rank by Score
        df = df.sort_values('score', ascending=False)
        
        # 7. Format Output
        result_lines = ["Prediction Ranking (Score = P^4 * Odds):"]
        result_lines.append(f"Context: {df.iloc[0]['weather'] if 'weather' in race_data[0] else ''} / {df.iloc[0]['distance'] if 'distance' in race_data[0] else ''}m")
        result_lines.append("-" * 40)
        
        for i, (_, row) in enumerate(df.iterrows()):
            symbol = "  "
            if i == 0: symbol = "◎ "
            elif i == 1: symbol = "○ "
            elif i == 2: symbol = "▲ "
            elif i == 3: symbol = "△ "
            
            line = f"{symbol} {i+1}. {row['name']} (Odds: {row['odds']}, Score: {row['score']:.4f})"
            result_lines.append(line)
            
        return "\n".join(result_lines)

    except Exception as e:
        import traceback
        return f"Prediction Error: {e}\n{traceback.format_exc()}"
