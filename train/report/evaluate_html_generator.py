
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
from train.features import FEATURES, PLACE_MAP_SHORT

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
    
    # min_scores はスコア生成後に動的に決定するため、ここではプレースホルダー
    min_scores = None  # スコア算出後に設定
    
    # 評価結果の格納
    result_summary = None

    # 1. Load Data & Model (Once)
    from train import scraper_bulk, preprocess
    import joblib
    
    if not os.path.exists(settings.MODEL_PATH):
        print("Model not found.")
        return

    print("Loading Model...")
    model = joblib.load(settings.MODEL_PATH)
    artifacts = joblib.load(os.path.join(settings.MODEL_DIR, 'encoders.pkl'))
    
    # Load Data
    print("Loading Data...")
    raw_df = preprocess.load_data(start_year=start_year, end_year=end_year, start_month=start_month, end_month=end_month)
    
    if raw_df.empty:
        print("No data found, skipping.")
        return

    # Filter places & races (CLI range filter)
    if not raw_df.empty:
        raw_df['race_id'] = raw_df['race_id'].astype(str)
        raw_df['race_no'] = raw_df['race_id'].str[-2:].astype(int)
        
        # CLI Range Filter (race_min/max)
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
    # transform() は未知カラムを通過させるため、変換後に行対応を保ったまま参照できる
    raw_df['_raw_rank'] = pd.to_numeric(raw_df['rank'], errors='coerce')
    raw_df['_raw_odds'] = pd.to_numeric(raw_df['odds'], errors='coerce').fillna(0)
    raw_df = raw_df.reset_index(drop=True)

    # Transform
    print("Transforming...")
    df_base = preprocess.transform(raw_df, artifacts)
    df_base = df_base.reset_index(drop=True)
    
    # Features — 共通定数を使用
    features = FEATURES
    
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
    # transformに持ち越した _raw_rank / _raw_odds を直接使用（行対応保証）
    df_base['rank'] = df_base['_raw_rank']
    df_base['odds'] = df_base['_raw_odds']
    
    # Pre-filtering for simulation
    df_base = df_base[df_base['place_code'].notna()]
    unique_places = sorted(df_base['place_code'].unique().astype(str))

    # スコア計算: LambdaRankスコア × オッズ
    # スコア = softmax確率（win_prob）のみ使用
    # 診断結果: win_prob のみで hit_rate 2.25%, win_prob×log(odds)では 0.96%
    df_base['score'] = df_base['win_prob']
    
    # min_scores をデータ実分布から動的生成
    # レース毎のtop1スコアの分布に基づいて 11段階の閾値を生成
    place_df_sorted_tmp = df_base.sort_values(['race_id', 'score'], ascending=[True, False])
    top1_scores = place_df_sorted_tmp.groupby('race_id')['score'].first()
    score_min = float(top1_scores.min())
    score_max = float(top1_scores.max())
    # 0〜max の範囲を10分割（0 は「フィルタなし」として常に含める）
    step = score_max / 10
    min_scores = [round(step * i, 1) for i in range(11)]
    print(f"スコアレンジ: {score_min:.1f} 〜 {score_max:.1f}、閾値: {min_scores}")
    
    summary_data = []
    
    for score_thresh in min_scores:
        # Group by Place
        for p_code in unique_places:
            place_name = place_map.get(p_code, f"Place {p_code}")
            place_df = df_base[df_base['place_code'] == p_code]
            
            # Get Top 1 per race
            place_df_sorted = place_df.sort_values(['race_id', 'score'], ascending=[True, False])
            top1_df = place_df_sorted.groupby('race_id').head(1)
            
            # Filter by threshold
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
            
            summary_data.append({
                'min_score': score_thresh,
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

    # C. Generate Report
    # 期間表示用の文字列を構築
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
        </style>
    </head>
    <body>
        <h1>Evaluation Report</h1>
        <p><strong>Evaluation Period:</strong> {eval_period}</p>
        <p><strong>Generated:</strong> {pd.Timestamp.now()}</p>
    """
    
    # 1. Performance Chart: ROI vs Score
    html_content += "<h2>ROI and Hit Rate by Score Threshold</h2>"
    
    plt.figure(figsize=(10, 6))
    
    agg = result_summary.groupby('min_score').agg({'bets': 'sum', 'cost': 'sum', 'return': 'sum', 'hits': 'sum'}).reset_index()
    agg['roi'] = (agg['return'] / agg['cost'] * 100).fillna(0)
    agg['hit_rate'] = (agg['hits'] / agg['bets'] * 100).fillna(0)
    
    plt.plot(agg['min_score'], agg['roi'], marker='o', label='ROI (%)')
    plt.plot(agg['min_score'], agg['hit_rate'], marker='x', label='Hit Rate (%)')
    
    plt.axhline(100, color='red', linestyle='--', label='Break Even (ROI)')
    plt.title("Performance vs Min Score Threshold")
    plt.xlabel("Min Score")
    plt.ylabel("Value (%)")
    plt.grid(True)
    plt.legend()
    
    best_configs = []
    valid_agg = agg[agg['bets'] >= 10]
    if not valid_agg.empty:
        best_row = valid_agg.loc[valid_agg['roi'].idxmax()]
        best_configs.append({
            'Best ROI': best_row['roi'],
            'At Score': best_row['min_score'],
            'Bets': best_row['bets']
        })
    
    buf = BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    data_uri = base64.b64encode(buf.getvalue()).decode('utf-8')
    html_content += f'<div class="chart"><img src="data:image/png;base64,{data_uri}" style="max-width:100%"></div>'
    
    # 1b. Feature Importance Chart (NEW)
    if 'feature_importance' in artifacts:
        html_content += "<h2>Feature Importance (Gain)</h2>"
        fi_df = pd.DataFrame(artifacts['feature_importance'])
        
        plt.figure(figsize=(10, 8))
        # Plot top 20
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
    
    # 2. Best Configuration Table
    html_content += "<h2>Best Configuration Summary (Min 10 bets)</h2>"
    if best_configs:
        best_df = pd.DataFrame(best_configs).sort_values('Best ROI', ascending=False)
        html_content += best_df.to_html(classes='table', float_format="%.2f", index=False)
    else:
        html_content += "<p>No configurations with >10 bets found.</p>"

    # 3. Detailed Metrics
    html_content += f"<h2>Detailed Metrics</h2>"
    
    # Overall by score
    agg = result_summary.groupby('min_score').agg({
        'bets': 'sum', 'hits': 'sum', 'cost': 'sum', 'return': 'sum', 'hits_top3': 'sum'
    }).reset_index()
    agg['roi'] = (agg['return'] / agg['cost'] * 100).fillna(0)
    agg['hit_rate'] = (agg['hits'] / agg['bets'] * 100).fillna(0)
    agg['place_rate'] = (agg['hits_top3'] / agg['bets'] * 100).fillna(0)
    
    cols = ['min_score', 'bets', 'hit_rate', 'place_rate', 'roi', 'return']
    html_content += f"<h3>Overall by Threshold</h3>"
    html_content += agg[cols].to_html(classes='table', float_format="%.2f", index=False)
    
    # By Place
    html_content += f"<h3>ROI by Racecourse</h3>"
    pivot_roi = result_summary.pivot_table(index='min_score', columns='place_name', values='roi', aggfunc='first')
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

