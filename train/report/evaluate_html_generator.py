
import pandas as pd
import numpy as np
import os
import argparse
import sys
import matplotlib.pyplot as plt
import base64
from io import BytesIO

# Add project root
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from train import settings
from train.features import FEATURES, PLACE_MAP_SHORT, compute_score

def generate_report(start_year, end_year, output_file="evaluate.html", race_min=None, race_max=None, start_month=None, end_month=None):
    if start_month and end_month:
        print(f"Generating Evaluation Report for {start_year}/{start_month}-{end_year}/{end_month}...")
    else:
        print(f"Generating Evaluation Report for {start_year}-{end_year}...")
    
    # Place Codes Mapping — 共通定数を使用
    place_map = PLACE_MAP_SHORT
    
    r_min = int(race_min) if race_min is not None else 1
    r_max = int(race_max) if race_max is not None else 12
    print(f"Evaluating Race Numbers: {r_min} to {r_max}")
    
    # 1. Load Data & Model (Once)
    from train import scraper_bulk, preprocess
    import joblib
    
    if not os.path.exists(settings.MODEL_PATH):
        print("Model not found.")
        return

    print("Loading Model...")
    model = joblib.load(settings.MODEL_PATH)
    artifacts = joblib.load(os.path.join(settings.MODEL_DIR, 'encoders.pkl'))
    
    # --- P2: 過去2年分のデータも合わせてロードし、lag特徴量を正確に計算 ---
    # 評価対象年だけでなく、前2年分もロードする
    history_start = start_year - 2
    print(f"Loading Data (history: {history_start}-{end_year}, eval: {start_year}-{end_year})...")
    raw_df = preprocess.load_data(start_year=history_start, end_year=end_year)
    
    if raw_df.empty:
        print("No data found, skipping.")
        return

    # Filter places & races (CLI range filter)
    if not raw_df.empty:
        raw_df['race_id'] = raw_df['race_id'].astype(str)
        raw_df['race_no'] = raw_df['race_id'].str[-2:].astype(int)
        
        # CLI Range Filter (race_min/max) — 全データに適用
        if race_min is not None or race_max is not None:
             print(f"Filtering race numbers by range: {r_min}-{r_max}")
             raw_df = raw_df[(raw_df['race_no'] >= r_min) & (raw_df['race_no'] <= r_max)]
        
        # JRA 10場のみに限定（地方競馬コード35/36/46/54/55等を除外）
        jra_codes = {f'{i:02d}' for i in range(1, 11)}
        raw_df['place_code_tmp'] = raw_df['race_id'].str[4:6]
        before_jra = len(raw_df['race_id'].unique())
        raw_df = raw_df[raw_df['place_code_tmp'].isin(jra_codes)]
        raw_df = raw_df.drop(columns=['place_code_tmp'])
        after_jra = len(raw_df['race_id'].unique())
        print(f"JRA フィルタ後: {after_jra}/{before_jra} レース")
             
    if raw_df.empty:
        print("No data after filtering.")
        return
    
    # rank・odds を transform に持ち越すため、一時カラムとして付与
    raw_df['_raw_rank'] = pd.to_numeric(raw_df['rank'], errors='coerce')
    raw_df['_raw_odds'] = pd.to_numeric(raw_df['odds'], errors='coerce').fillna(0)
    # year カラムを保持（後でフィルタに使用）
    raw_df['_raw_year'] = pd.to_numeric(raw_df['year'], errors='coerce')
    raw_df['_raw_month'] = pd.to_numeric(raw_df['month'], errors='coerce') if 'month' in raw_df.columns else 0
    raw_df = raw_df.reset_index(drop=True)

    # Transform（全データでlag特徴量を計算）
    print("Transforming (with historical context)...")
    df_base = preprocess.transform(raw_df, artifacts)
    df_base = df_base.reset_index(drop=True)
    
    # --- P2: 評価対象期間でフィルタ（lag特徴量計算後） ---
    df_base['_eval_year'] = df_base['_raw_year'] if '_raw_year' in df_base.columns else pd.to_numeric(df_base.get('year', 0), errors='coerce')
    df_base['_eval_month'] = df_base['_raw_month'] if '_raw_month' in df_base.columns else pd.to_numeric(df_base.get('month', 0), errors='coerce')
    
    # 評価対象期間でフィルタ
    year_mask = (df_base['_eval_year'] >= start_year) & (df_base['_eval_year'] <= end_year)
    if start_month and end_month:
        month_mask = (df_base['_eval_month'] >= start_month) & (df_base['_eval_month'] <= end_month)
        eval_mask = year_mask & month_mask
    else:
        eval_mask = year_mask
    
    rows_before = len(df_base)
    df_base = df_base[eval_mask].reset_index(drop=True)
    print(f"評価対象フィルタ: {rows_before} → {len(df_base)} 行 (eval: {start_year}/{start_month or 1}-{end_year}/{end_month or 12})")
    
    if df_base.empty:
        print("No evaluation data after filtering.")
        return
    
    # 3. Predict (Raw logits)
    df_base['win_prob_raw'] = model.predict(df_base[FEATURES])
    df_base['race_id_raw'] = df_base['race_id'].astype(str) if 'race_id' in df_base.columns else raw_df['race_id'].astype(str)
    
    # レース単位の softmax
    def softmax(group):
        exps = np.exp(group - group.max())
        return exps / exps.sum()
    df_base['win_prob'] = df_base.groupby('race_id_raw')['win_prob_raw'].transform(softmax)
        
    # Attach Metadata
    df_base['race_id'] = df_base['race_id_raw']
    df_base['place_code'] = df_base['race_id'].str[4:6]
    df_base['rank'] = df_base['_raw_rank']
    df_base['odds'] = df_base['_raw_odds']
    
    # Pre-filtering for simulation
    df_base = df_base[df_base['place_code'].notna()]
    unique_places = sorted(df_base['place_code'].unique().astype(str))

    # score（期待値）も計算（参考用）
    df_base['score'] = compute_score(df_base, win_prob_col='win_prob', odds_col='odds')
    
    # --- P1: win_prob ベースの閾値を動的生成 ---
    # Top-1をwin_probで選択し、閾値をwin_probの分布から決定
    prob_sorted_tmp = df_base.sort_values(['race_id', 'win_prob'], ascending=[True, False])
    top1_probs = prob_sorted_tmp.groupby('race_id')['win_prob'].first()
    prob_min = float(top1_probs.min())
    prob_max = float(top1_probs.max())
    # 0.0 から prob_max まで 11段階の閾値
    prob_step = prob_max / 10 if prob_max > 0 else 0.05
    min_probs = [round(prob_step * i, 4) for i in range(11)]
    print(f"Win_prob レンジ: {prob_min:.4f} 〜 {prob_max:.4f}、閾値: {min_probs}")
    
    # =============================================================
    # A. Win_Prob ベースの評価（P1: hit_rate 最大化）
    # =============================================================
    summary_data = []
    
    for prob_thresh in min_probs:
        for p_code in unique_places:
            place_name = place_map.get(p_code, f"Place {p_code}")
            place_df = df_base[df_base['place_code'] == p_code]
            
            # --- P1: Top 1 を win_prob でソート ---
            place_df_sorted = place_df.sort_values(['race_id', 'win_prob'], ascending=[True, False])
            top1_df = place_df_sorted.groupby('race_id').head(1)
            
            # win_prob 閾値でフィルタ
            bet_df = top1_df[top1_df['win_prob'] >= prob_thresh]
            
            bets = len(bet_df)
            if bets > 0:
                cost = bets * 100
                hits_df = bet_df[bet_df['rank'] == 1]
                hits = len(hits_df)
                place_df_hits = bet_df[bet_df['rank'] <= 3]
                hits_top3 = len(place_df_hits)
                return_amt = (hits_df['odds'] * 100).sum()
                
                roi = return_amt / cost * 100
                hit_rate = hits / bets * 100
                place_rate = hits_top3 / bets * 100
            else:
                bets, hits, hits_top3, return_amt, cost = 0, 0, 0, 0, 0
                roi, hit_rate, place_rate = 0, 0, 0
            
            summary_data.append({
                'min_prob': prob_thresh,
                'place_code': p_code,
                'place_name': place_name,
                'bets': bets,
                'hits': hits,
                'hits_top3': hits_top3,
                'hit_rate': hit_rate,
                'place_rate': place_rate,
                'roi': roi,
                'return': return_amt,
                'cost': cost
            })
    
    result_summary = pd.DataFrame(summary_data)

    # =============================================================
    # B. Score (win_prob×odds) ベースの評価（参考: ROI最適化）
    # =============================================================
    score_sorted_tmp = df_base.sort_values(['race_id', 'score'], ascending=[True, False])
    top1_scores = score_sorted_tmp.groupby('race_id')['score'].first()
    score_min = float(top1_scores.min())
    score_max = float(top1_scores.max())
    score_step = score_max / 10 if score_max > 0 else 0.1
    min_scores = [round(score_step * i, 1) for i in range(11)]
    
    score_summary_data = []
    for score_thresh in min_scores:
        for p_code in unique_places:
            place_name = place_map.get(p_code, f"Place {p_code}")
            place_df = df_base[df_base['place_code'] == p_code]
            
            place_df_sorted = place_df.sort_values(['race_id', 'score'], ascending=[True, False])
            top1_df = place_df_sorted.groupby('race_id').head(1)
            bet_df = top1_df[top1_df['score'] >= score_thresh]
            
            bets = len(bet_df)
            if bets > 0:
                cost = bets * 100
                hits_df = bet_df[bet_df['rank'] == 1]
                hits = len(hits_df)
                place_df_hits = bet_df[bet_df['rank'] <= 3]
                hits_top3 = len(place_df_hits)
                return_amt = (hits_df['odds'] * 100).sum()
                roi = return_amt / cost * 100
                hit_rate = hits / bets * 100
                place_rate = hits_top3 / bets * 100
            else:
                bets, hits, hits_top3, return_amt, cost = 0, 0, 0, 0, 0
                roi, hit_rate, place_rate = 0, 0, 0
            
            score_summary_data.append({
                'min_score': score_thresh,
                'place_code': p_code,
                'place_name': place_name,
                'bets': bets, 'hits': hits, 'hits_top3': hits_top3,
                'hit_rate': hit_rate, 'place_rate': place_rate,
                'roi': roi, 'return': return_amt, 'cost': cost
            })
    
    score_result_summary = pd.DataFrame(score_summary_data)

    # =============================================================
    # C. Generate HTML Report
    # =============================================================
    if start_month and end_month:
        eval_period = f"{start_year}/{start_month:02d} - {end_year}/{end_month:02d}"
        title_period = f"{start_year}/{start_month:02d}-{end_year}/{end_month:02d}"
    else:
        eval_period = f"{start_year} - {end_year}"
        title_period = f"{start_year}-{end_year}"
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Evaluation Report ({title_period})</title>
        <style>
            body {{ font-family: sans-serif; margin: 20px; }}
            table {{ border-collapse: collapse; width: 100%; margin-bottom: 30px; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
            h2 {{ border-bottom: 2px solid #333; padding-bottom: 5px; }}
            .chart {{ margin: 20px 0; border: 1px solid #eee; padding: 10px; }}
            .container {{ display: flex; flex-wrap: wrap; }}
            .box {{ margin-right: 20px; }}
            .section-prob {{ border-left: 4px solid #2196F3; padding-left: 12px; margin-bottom: 30px; }}
            .section-score {{ border-left: 4px solid #FF9800; padding-left: 12px; margin-bottom: 30px; }}
        </style>
    </head>
    <body>
        <h1>Evaluation Report</h1>
        <p><strong>Evaluation Period:</strong> {eval_period}</p>
        <p><strong>Generated:</strong> {pd.Timestamp.now()}</p>
    """
    
    # --- Chart 1: Win_Prob ベース（メイン） ---
    html_content += '<div class="section-prob">'
    html_content += "<h2>📊 Win Probability Based (Hit Rate Optimized)</h2>"
    
    plt.figure(figsize=(10, 6))
    
    agg = result_summary.groupby('min_prob').agg({'bets': 'sum', 'cost': 'sum', 'return': 'sum', 'hits': 'sum', 'hits_top3': 'sum'}).reset_index()
    agg['roi'] = (agg['return'] / agg['cost'] * 100).fillna(0)
    agg['hit_rate'] = (agg['hits'] / agg['bets'] * 100).fillna(0)
    agg['place_rate'] = (agg['hits_top3'] / agg['bets'] * 100).fillna(0)
    
    plt.plot(agg['min_prob'], agg['roi'], marker='o', label='ROI (%)', color='#2196F3')
    plt.plot(agg['min_prob'], agg['hit_rate'], marker='x', label='Hit Rate (%)', color='#4CAF50')
    plt.plot(agg['min_prob'], agg['place_rate'], marker='s', label='Place Rate (%)', color='#FF9800')
    
    plt.axhline(100, color='red', linestyle='--', label='Break Even (ROI)')
    plt.title("Performance vs Min Win Probability Threshold")
    plt.xlabel("Min Win Probability")
    plt.ylabel("Value (%)")
    plt.grid(True)
    plt.legend()
    
    buf = BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    data_uri = base64.b64encode(buf.getvalue()).decode('utf-8')
    html_content += f'<div class="chart"><img src="data:image/png;base64,{data_uri}" style="max-width:100%"></div>'
    
    # Best Config (Win_Prob)
    best_configs = []
    valid_agg = agg[agg['bets'] >= 10]
    if not valid_agg.empty:
        # Best Hit Rate
        best_hr = valid_agg.loc[valid_agg['hit_rate'].idxmax()]
        best_configs.append({'Metric': 'Best Hit Rate', 'Value': f"{best_hr['hit_rate']:.2f}%", 'At Prob': best_hr['min_prob'], 'Bets': int(best_hr['bets']), 'ROI': f"{best_hr['roi']:.2f}%"})
        # Best ROI
        best_roi = valid_agg.loc[valid_agg['roi'].idxmax()]
        best_configs.append({'Metric': 'Best ROI', 'Value': f"{best_roi['roi']:.2f}%", 'At Prob': best_roi['min_prob'], 'Bets': int(best_roi['bets']), 'Hit Rate': f"{best_roi['hit_rate']:.2f}%"})
    
    html_content += "<h3>Best Configuration (Min 10 bets)</h3>"
    if best_configs:
        best_df = pd.DataFrame(best_configs)
        html_content += best_df.to_html(classes='table', index=False, na_rep="-")
    else:
        html_content += "<p>No configurations with >10 bets found.</p>"

    # Detailed Table (Win_Prob)
    cols = ['min_prob', 'bets', 'hit_rate', 'place_rate', 'roi', 'return']
    html_content += "<h3>Overall by Win Probability Threshold</h3>"
    html_content += agg[cols].to_html(classes='table', float_format="%.2f", index=False)
    
    # By Place (Win_Prob)
    html_content += "<h3>Hit Rate by Racecourse</h3>"
    pivot_hr = result_summary.pivot_table(index='min_prob', columns='place_name', values='hit_rate', aggfunc='first')
    html_content += pivot_hr.to_html(classes='table', float_format="%.1f%%", na_rep="-")
    
    html_content += '</div>'  # end section-prob
    
    # --- Chart 2: Score (Expected Value) ベース（参考） ---
    html_content += '<div class="section-score">'
    html_content += "<h2>📈 Expected Value Based (ROI Optimized, Reference)</h2>"
    
    plt.figure(figsize=(10, 6))
    
    score_agg = score_result_summary.groupby('min_score').agg({'bets': 'sum', 'cost': 'sum', 'return': 'sum', 'hits': 'sum', 'hits_top3': 'sum'}).reset_index()
    score_agg['roi'] = (score_agg['return'] / score_agg['cost'] * 100).fillna(0)
    score_agg['hit_rate'] = (score_agg['hits'] / score_agg['bets'] * 100).fillna(0)
    score_agg['place_rate'] = (score_agg['hits_top3'] / score_agg['bets'] * 100).fillna(0)
    
    plt.plot(score_agg['min_score'], score_agg['roi'], marker='o', label='ROI (%)', color='#FF9800')
    plt.plot(score_agg['min_score'], score_agg['hit_rate'], marker='x', label='Hit Rate (%)', color='#4CAF50')
    
    plt.axhline(100, color='red', linestyle='--', label='Break Even (ROI)')
    plt.title("Performance vs Min Score Threshold (Expected Value)")
    plt.xlabel("Min Score (win_prob × odds)")
    plt.ylabel("Value (%)")
    plt.grid(True)
    plt.legend()
    
    buf2 = BytesIO()
    plt.savefig(buf2, format='png')
    plt.close()
    score_uri = base64.b64encode(buf2.getvalue()).decode('utf-8')
    html_content += f'<div class="chart"><img src="data:image/png;base64,{score_uri}" style="max-width:100%"></div>'

    score_cols = ['min_score', 'bets', 'hit_rate', 'place_rate', 'roi', 'return']
    html_content += "<h3>Overall by Score Threshold</h3>"
    html_content += score_agg[score_cols].to_html(classes='table', float_format="%.2f", index=False)
    
    html_content += '</div>'  # end section-score
    
    # --- Feature Importance Chart ---
    if 'feature_importance' in artifacts:
        html_content += "<h2>Feature Importance (Gain)</h2>"
        fi_df = pd.DataFrame(artifacts['feature_importance'])
        
        plt.figure(figsize=(10, 8))
        top_fi = fi_df.head(20).sort_values('importance', ascending=True)
        plt.barh(top_fi['feature'], top_fi['importance'], color='skyblue')
        plt.title("LightGBM Feature Importance (Gain)")
        plt.xlabel("Total Gain")
        plt.tight_layout()
        
        buf_fi = BytesIO()
        plt.savefig(buf_fi, format='png')
        plt.close()
        fi_uri = base64.b64encode(buf_fi.getvalue()).decode('utf-8')
        html_content += f'<div class="chart"><img src="data:image/png;base64,{fi_uri}" style="max-width:100%"></div>'
    else:
        html_content += "<h2>Feature Importance</h2><p>Feature importance data not found in artifacts. Re-train the model to generate this data.</p>"
    
    # --- ROI by Racecourse (Score-based, reference) ---
    html_content += "<h2>Detailed Metrics</h2>"
    html_content += "<h3>ROI by Racecourse (Score-based)</h3>"
    pivot_roi = score_result_summary.pivot_table(index='min_score', columns='place_name', values='roi', aggfunc='first')
    html_content += pivot_roi.to_html(classes='table', float_format="%.1f%%", na_rep="-")

    html_content += "</body></html>"
    
    with open(output_file, "w", encoding='utf-8') as f:
        f.write(html_content)
        
    print(f"Report saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, default=2025)
    parser.add_argument("--end", type=int, default=2025)
    parser.add_argument("--output", type=str, default="evaluate.html")
    parser.add_argument("--race_min", type=int, default=None, help="Min Race No")
    parser.add_argument("--race_max", type=int, default=None, help="Max Race No")
    parser.add_argument("--start_month", type=int, default=None, help="Start Month (1-12)")
    parser.add_argument("--end_month", type=int, default=None, help="End Month (1-12)")
    args = parser.parse_args()
    
    generate_report(args.start, args.end, args.output, args.race_min, args.race_max, args.start_month, args.end_month)

