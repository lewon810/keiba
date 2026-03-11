import joblib
import pandas as pd
import os
import sys

# プロジェクトルートをパスに追加
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from train import settings
    from train.features import FEATURES, lookup_rate, get_dist_cat, apply_label_encoder
except ImportError:
    class settings:
        MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'train', 'data', 'model')
        MODEL_PATH = os.path.join(MODEL_DIR, 'model_lgb.pkl')
    FEATURES = None
    lookup_rate = None
    get_dist_cat = None
    apply_label_encoder = None

def predict(race_data, return_df=False):
    """
    Takes race data (list of dicts) and returns predictions using the trained model.
    If return_df is True, returns the pandas DataFrame with win probability.
    Predictions are ranked by win_prob.
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
                    df.at[i, 'lag1_last_3f'] = last_stats['lag1_last_3f']
                    df.at[i, 'interval'] = last_stats['interval']
                else:
                    df.at[i, 'lag1_rank'] = 99
                    df.at[i, 'lag1_speed_index'] = 0
                    df.at[i, 'lag1_last_3f'] = 0
                    df.at[i, 'interval'] = 365
                
                # Lag 2, 3 — 2走前・3走前のデータ
                lag2_stats = loader.get_nth_last_race(row['horse_id'], n=2, current_date_str=current_date) if hasattr(loader, 'get_nth_last_race') else None
                lag3_stats = loader.get_nth_last_race(row['horse_id'], n=3, current_date_str=current_date) if hasattr(loader, 'get_nth_last_race') else None
                df.at[i, 'lag2_rank'] = lag2_stats['lag1_rank'] if lag2_stats else 99
                df.at[i, 'lag3_rank'] = lag3_stats['lag1_rank'] if lag3_stats else 99

        except Exception as e:
            print(f"⚠️  History load failed: {e}")
            print("⚠️  Using default feature values - prediction accuracy will be reduced.")
            df['lag1_rank'] = 99
            df['lag1_speed_index'] = 0
            df['lag1_last_3f'] = 0
            df['interval'] = 365
            df['lag2_rank'] = 99
            df['lag3_rank'] = 99
            df['lag2_speed_index'] = 0
            df['avg_last3_speed_index'] = 0
            df['speed_trend'] = 0
        
        # 直近3走の平均着順
        df['avg_last3_rank'] = df[['lag1_rank', 'lag2_rank', 'lag3_rank']].mean(axis=1)

        # 特徴量生成前の事前処理 (欠損カテゴリの補完)
        cat_cols = ['horse_id', 'jockey_id', 'trainer_id', 'course_type', 'weather', 'condition', 'sire_id', 'damsire_id', 'running_style']
        for col in cat_cols:
            if col not in df.columns:
                 df[col] = "unknown"

        # スピード指数トレンド特有のフォールバック (履歴がない場合)
        if 'lag2_speed_index' not in df.columns:
            df['lag2_speed_index'] = 0
        if 'avg_last3_speed_index' not in df.columns:
            if 'lag1_speed_index' in df.columns:
                df['avg_last3_speed_index'] = df['lag1_speed_index']
            else:
                df['avg_last3_speed_index'] = 0
        if 'speed_trend' not in df.columns:
            df['speed_trend'] = 0

        # 共通関数で artifacts (勝率マップ, LabelEncoder) を使った特徴量を一括適用
        from train.features import apply_artifacts_to_df
        df = apply_artifacts_to_df(df, artifacts)

        # ... (Numeric cleanup skipped in this diff, assuming follow-up or inclusion)
        # 4. Numeric cleanup
        df['waku'] = pd.to_numeric(df['waku'], errors='coerce').fillna(0)
        df['umaban'] = pd.to_numeric(df['umaban'], errors='coerce').fillna(0)
        df['distance'] = pd.to_numeric(df['distance'], errors='coerce').fillna(0)
        
        # Missing columns handling
        if 'weight_diff' not in df.columns:
            df['weight_diff'] = 0
        df['weight_diff'] = pd.to_numeric(df['weight_diff'], errors='coerce').fillna(0)

        # Feature: Popularity（人気順位）
        if 'popularity' in df.columns:
            df['popularity'] = pd.to_numeric(df['popularity'], errors='coerce').fillna(99)
        else:
            df['popularity'] = 99
        
        # Feature: Number of Runners（出走頭数）
        df['num_runners'] = len(df)
        
        # Feature: Horse Age（馬齢）
        if 'horse_id' in df.columns:
            import datetime
            current_year = datetime.datetime.now().year
            df['horse_birth_year'] = df['horse_id'].astype(str).str[:4]
            df['horse_birth_year'] = pd.to_numeric(df['horse_birth_year'], errors='coerce')
            df['horse_age'] = current_year - df['horse_birth_year']
            df['horse_age'] = df['horse_age'].clip(lower=2, upper=10).fillna(3)
            df = df.drop(columns=['horse_birth_year'], errors='ignore')
        else:
            df['horse_age'] = 3
        
        # Running Style — デフォルトは unknown（推論時は前走データなし）
        if 'running_style' not in df.columns:
            df['running_style'] = "unknown"

        # 5. Predict — 共通定数を使用
        features = FEATURES

        missing_cols = set(features) - set(df.columns)
        if missing_cols:
            print(f"DEBUG Missing columns: {missing_cols}")

        # LambdaRank returns 1D score array (N,) - higher is better
        pred_scores = model.predict(df[features])
        df['pred_scores'] = pred_scores
        
        # Convert LambdaRank scores to probabilities using softmax
        # This prevents the top horse from always being 100% and creates a realistic probability distribution
        import numpy as np
        
        # Clean Odds for calculation
        def parse_odds(o):
            try:
                return float(o)
            except:
                return 0.0
        df['odds_val'] = df['odds'].apply(parse_odds)

        if len(pred_scores) > 0:
            # 1. Softmax transformation for numerical stability (Base Probability)
            exp_scores = np.exp(pred_scores - pred_scores.max())
            base_prob = exp_scores / exp_scores.sum()
            
            # 2. Implied Probability from Market Odds (assuming 80% payout rate)
            # odds_val が 0 の場合は 0.0 とする
            implied_prob = 0.80 / df['odds_val'].replace(0.0, np.nan)
            implied_prob = implied_prob.fillna(0.0)
            
            # 3. Hybrid Blend (Geometric Mean)
            # 市場が全く評価していない（オッズが高い）馬でもベース能力が高ければある程度の確率が残るが、
            # ベース単体よりは現実的な水準（一桁台前半）に抑えられる。
            # ※ odds_val==0 (オッズ取得失敗) の場合は base_prob をそのまま使う
            df['win_prob'] = np.where(
                df['odds_val'] > 0,
                np.sqrt(base_prob * implied_prob),
                base_prob
            )
            # 正規化して合計を1.0にする
            df['win_prob'] = df['win_prob'] / df['win_prob'].sum()
        else:
            df['win_prob'] = 0.0

        # スコア計算: 共通関数を使用（ハイブリッド勝率 × オッズ = 期待値）
        # scoreロジック（黒歴史）は削除されました

        # Rank by Win Probability (Descending)
        df = df.sort_values('win_prob', ascending=False)
        
        if return_df:
            return df

        # 6. Format Output
        # Get context from original race_data to avoid showing encoded integers
        context_weather = race_data[0].get('weather', 'Unknown')
        context_distance = race_data[0].get('distance', 'Unknown')

        result_lines = ["Prediction Ranking (Ranked by Win Probability):"]
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
            
            line = f"{symbol} {i+1}. {row['name']} (Odds: {odds_str}, Win%: {prob_pct:.1f}%)"
            result_lines.append(line)

        return "\n".join(result_lines)

    except Exception as e:
        import traceback
        return f"Prediction Error: {e}\n{traceback.format_exc()}"