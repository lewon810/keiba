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

from train import evaluate
from train import settings

def generate_report(start_year, end_year, output_file="evaluate.html"):
    print(f"Generating Evaluation Report for {start_year}-{end_year}...")
    
    # Range of scores to test
    min_scores = [round(x * 0.1, 1) for x in range(0, 11)] # 0.0 to 1.0
    
    # Store results
    # Structure: {place_name: {min_score: {roi: ..., hit_rate: ..., bets: ...}}}
    results = {}
    
    # Place Codes Mapping
    place_map = {
        "01": "Sapporo", "02": "Hakodate", "03": "Fukushima", "04": "Niigata",
        "05": "Tokyo", "06": "Nakayama", "07": "Chukyo", "08": "Kyoto",
        "09": "Hanshin", "10": "Kokura"
    }
    
    # 1. Run Evaluation for each score threshold
    # To save time, we should load data ONCE and then filter in memory if possible.
    # evaluate.evaluate function performs loading -> predict -> simulate.
    # We can refactor evaluate.py to separate prediction and simulation, 
    # OR just call it repeatedly (easier but slower due to model loading).
    # Given the requirements, let's try to capture the PREDICTIONS from evaluate.py logic?
    # actually, evaluate.py is designed to print.
    # Let's import the core logic or modify evaluate.py to return the DF with scores.
    
    # Let's effectively "rewrite" the simulation loop here using one-time prediction
    
    # A. Load & Predict (Once)
    # Re-use evaluate logic manually
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
    raw_df = pd.DataFrame()
    # Try year by year
    dfs = []
    for y in range(start_year, end_year + 1):
        p = os.path.join(settings.RAW_DATA_DIR, f"results_{y}.csv")
        if os.path.exists(p):
            dfs.append(pd.read_csv(p))
            
    if dfs:
        raw_df = pd.concat(dfs, ignore_index=True)
    else:
        # scraping is disabled per user rule for this step or we assume it exists.
        print("No data found, skipping.")
        return

    # Load Filters
    import yaml
    yaml_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'evaluate_settings.yml')
    config = {}
    if os.path.exists(yaml_path):
        with open(yaml_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
            
    # Filter places & races
    if not raw_df.empty and config:
        raw_df['race_id'] = raw_df['race_id'].astype(str)
        
        # Place Filter
        target_places = config.get('target_places', [])
        if target_places:
            print(f"Filtering places: {target_places}")
            # Place code is 4:6 e.g. 2023 06 ...
            raw_df = raw_df[raw_df['race_id'].str[4:6].isin([str(p).zfill(2) for p in target_places])]
            
        # Race No Filter
        target_races = config.get('target_race_numbers', [])
        if target_races:
            print(f"Filtering race numbers: {target_races}")
            raw_df['race_no'] = raw_df['race_id'].str[-2:].astype(int)
            raw_df = raw_df[raw_df['race_no'].isin(target_races)]
            
    if raw_df.empty:
        print("No data after filtering.")
        return
    
    # Transform
    print("Transforming...")
    df = preprocess.transform(raw_df, artifacts)
    
    # Features (Sync with train.py)
    features = [
        'jockey_win_rate', 'trainer_win_rate', 'horse_id', 'jockey_id', 'trainer_id',
        'waku', 'umaban', 'course_type', 'distance', 'weather', 'condition',
        'lag1_rank', 'lag1_speed_index', 'interval', 'weight_diff'
    ]
    
    print("Predicting...")
    pred_probs = model.predict(df[features])
    if pred_probs.ndim > 1:
        df['win_prob'] = pred_probs[:, 0]
    else:
        df['win_prob'] = pred_probs
        
    # Attach Metadata
    df['race_id'] = raw_df['race_id'].astype(str)
    df['place_code'] = df['race_id'].str[4:6]
    df['rank'] = pd.to_numeric(raw_df['rank'], errors='coerce')
    df['odds'] = pd.to_numeric(raw_df['odds'], errors='coerce').fillna(0)
    
    # Calculate Score
    df['score'] = (df['win_prob'] ** 4) * df['odds']
    
    # B. Simulation Loop
    summary_data = []
    detailed_logs = []
    
    # Ensure place_code is valid
    df = df[df['place_code'].notna()]
    unique_places = sorted(df['place_code'].unique().astype(str))
    
    print("Simulating ROI for thresholds...")
    for score_thresh in min_scores:
        # Group by Place
        for p_code in unique_places:
            place_name = place_map.get(p_code, f"Place {p_code}")
            
            # Filter rows for this place
            place_df = df[df['place_code'] == p_code]
            
            # Group by Race
            # We need to pick Top 1 per race
            # Efficient Grouping
            # We only care about the horse with max score in each race
            # AND if that score >= thresh
            
            # 1. Get Top 1 per race
            # sort by race_id, score desc
            place_df_sorted = place_df.sort_values(['race_id', 'score'], ascending=[True, False])
            top1_df = place_df_sorted.groupby('race_id').head(1)
            
            # 2. Filter by threshold
            bet_df = top1_df[top1_df['score'] >= score_thresh]
            
            # 3. Calc Metrics
            bets = len(bet_df)
            if bets > 0:
                cost = bets * 100
                
                # Hits (Rank 1)
                hits_df = bet_df[bet_df['rank'] == 1]
                hits = len(hits_df)
                
                # Top 3 (Place)
                place_df_hits = bet_df[bet_df['rank'] <= 3]
                hits_top3 = len(place_df_hits)
                
                # Return
                return_amt = (hits_df['odds'] * 100).sum()
                
                roi = return_amt / cost * 100
                hit_rate = hits / bets * 100
                place_rate = hits_top3 / bets * 100
                
                # Logging (Only need to log once per race, but we are inside loop over thresholds.
                # Let's log if threshold is specific, or just log all bets for the lowest threshold (0.0) 
                # and let analysis filter? 
                # Or better: Create a separate log list for the specific requested investigation?
                # The user asked to analyze "min_score=0.1". 
                # Let's log EVERYTHING for min_score=0.1 specifically.
                
                if score_thresh == 0.1:
                    for _, row in bet_df.iterrows():
                        detailed_logs.append({
                            'race_id': row['race_id'],
                            'place': place_name,
                            'horse_id': row['horse_id'],
                            'score': float(row['score']),
                            'odds': float(row['odds']),
                            'rank': int(row['rank']) if pd.notnull(row['rank']) else -1,
                            'is_hit': 1 if row['rank'] == 1 else 0,
                            'return': float(row['odds'] * 100) if row['rank'] == 1 else 0
                        })

            else:
                roi = 0
                hit_rate = 0
                place_rate = 0
                hits_top3 = 0
            
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

    # Save Log
    import json
    log_file = output_file.replace(".html", "_log.json")
    with open(log_file, "w", encoding='utf-8') as f:
        json.dump(detailed_logs, f, indent=2, ensure_ascii=False)
    print(f"Log saved to {log_file}")

    # C. Generate PDF/HTML
    summary_df = pd.DataFrame(summary_data)
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Evaluation Report ({start_year}-{end_year})</title>
        <style>
            body {{ font-family: sans-serif; margin: 20px; }}
            table {{ border-collapse: collapse; width: 100%; margin-bottom: 30px; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
            h2 {{ border-bottom: 2px solid #333; padding-bottom: 5px; }}
            .chart {{ margin: 20px 0; border: 1px solid #eee; padding: 10px; }}
        </style>
    </head>
    <body>
        <h1>Evaluation Report</h1>
        <p><strong>Period:</strong> {start_year} - {end_year}</p>
        <p><strong>Generated:</strong> {pd.Timestamp.now()}</p>
    """
    
    # 1. Overall Summary Table (Pivot by Score)
    # Aggregate across all places
    all_summary = summary_df.groupby('min_score').agg({
        'bets': 'sum', 'hits': 'sum', 'hits_top3': 'sum', 'cost': 'sum', 'return': 'sum'
    }).reset_index()
    all_summary['roi'] = (all_summary['return'] / all_summary['cost'] * 100).fillna(0)
    all_summary['hit_rate'] = (all_summary['hits'] / all_summary['bets'] * 100).fillna(0)
    all_summary['place_rate'] = (all_summary['hits_top3'] / all_summary['bets'] * 100).fillna(0)
    
    html_content += "<h2>Overall Performance by Score Threshold</h2>"
    # Reorder cols for clarity
    cols = ['min_score', 'bets', 'hits', 'hits_top3', 'hit_rate', 'place_rate', 'roi', 'cost', 'return']
    html_content += all_summary[cols].to_html(classes='table', float_format="%.2f", index=False)
    
    # 2. ROI by Place (for each threshold)
    html_content += "<h2>ROI by Racecourse</h2>"
    
    pivot_roi = summary_df.pivot(index='min_score', columns='place_name', values='roi')
    html_content += pivot_roi.to_html(classes='table', float_format="%.1f%%", na_rep="-")

    # 3. Hit Rate by Place
    html_content += "<h2>Hit Rate (Win) by Racecourse</h2>"
    pivot_acc = summary_df.pivot(index='min_score', columns='place_name', values='hit_rate')
    html_content += pivot_acc.to_html(classes='table', float_format="%.1f%%", na_rep="-")
    
    # 4. Place Rate by Place
    html_content += "<h2>Place Rate (Top 3) by Racecourse</h2>"
    pivot_place = summary_df.pivot(index='min_score', columns='place_name', values='place_rate')
    html_content += pivot_place.to_html(classes='table', float_format="%.1f%%", na_rep="-")
    
    # 5. Charts (Matplotlib -> Base64)
    html_content += "<h2>Visual Analysis</h2>"
    
    # Chart 1: ROI vs Score (All)
    plt.figure(figsize=(10, 5))
    plt.plot(all_summary['min_score'], all_summary['roi'], marker='o', label='ROI')
    plt.axhline(100, color='red', linestyle='--', label='Break Even')
    plt.title("ROI vs Min Score Threshold")
    plt.xlabel("Min Score")
    plt.ylabel("ROI (%)")
    plt.grid(True)
    plt.legend()
    
    buf = BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    data_uri = base64.b64encode(buf.getvalue()).decode('utf-8')
    html_content += f'<div class="chart"><img src="data:image/png;base64,{data_uri}" style="max-width:100%"></div>'

    html_content += "</body></html>"
    
    with open(output_file, "w", encoding='utf-8') as f:
        f.write(html_content)
        
    print(f"Report saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, default=2025)
    parser.add_argument("--end", type=int, default=2025)
    parser.add_argument("--output", type=str, default="evaluate.html")
    args = parser.parse_args()
    
    generate_report(args.start, args.end, args.output)
