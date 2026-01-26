
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

def generate_report(start_year, end_year, output_file="evaluate.html", power_min=None, power_max=None, race_min=None, race_max=None):
    print(f"Generating Evaluation Report for {start_year}-{end_year}...")
    
    # ... (Defaults for power skipped in this diff, assuming context handles it or I include it)
    # Re-stating defaults to be safe with replace
    p_min = int(power_min) if power_min is not None else settings.POWER_EXPONENT
    p_max = int(power_max) if power_max is not None else p_min
    power_values = list(range(p_min, p_max + 1))
    
    r_min = int(race_min) if race_min is not None else 1
    r_max = int(race_max) if race_max is not None else 12
    print(f"Evaluating Power Exponents: {power_values}")
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
    
    # Load Data
    print("Loading Data...")
    raw_df = pd.DataFrame()
    dfs = []
    for y in range(start_year, end_year + 1):
        p = os.path.join(settings.RAW_DATA_DIR, f"results_{y}.csv")
        if os.path.exists(p):
            dfs.append(pd.read_csv(p))
    
    if dfs:
        raw_df = pd.concat(dfs, ignore_index=True)
    else:
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
    if not raw_df.empty:
        raw_df['race_id'] = raw_df['race_id'].astype(str)
        raw_df['race_no'] = raw_df['race_id'].str[-2:].astype(int)
        
        # 1. Config Filters (evaluate_settings.yml)
        if config:
            target_places = config.get('target_places', [])
            if target_places:
                print(f"Filtering places: {target_places}")
                raw_df = raw_df[raw_df['race_id'].str[4:6].isin([str(p).zfill(2) for p in target_places])]
            
            target_races = config.get('target_race_numbers', [])
            if target_races:
                print(f"Filtering race numbers from settings: {target_races}")
                raw_df = raw_df[raw_df['race_no'].isin(target_races)]
        
        # 2. CLI Range Filter (race_min/max)
        if race_min is not None or race_max is not None:
             print(f"Filtering race numbers by range: {r_min}-{r_max}")
             raw_df = raw_df[(raw_df['race_no'] >= r_min) & (raw_df['race_no'] <= r_max)]
             
    if raw_df.empty:
        print("No data after filtering.")
        return
    
    # Transform
    print("Transforming...")
    df_base = preprocess.transform(raw_df, artifacts)
    
    # Features
    features = [
        'jockey_win_rate', 'trainer_win_rate', 'horse_id', 'jockey_id', 'trainer_id',
        'waku', 'umaban', 'course_type', 'distance', 'weather', 'condition',
        'lag1_rank', 'lag1_speed_index', 'interval', 'weight_diff'
    ]
    
    print("Predicting...")
    pred_probs = model.predict(df_base[features])
    if pred_probs.ndim > 1:
        df_base['win_prob'] = pred_probs[:, 0]
    else:
        df_base['win_prob'] = pred_probs
        
    # Attach Metadata
    df_base['race_id'] = raw_df['race_id'].astype(str)
    df_base['place_code'] = df_base['race_id'].str[4:6]
    df_base['rank'] = pd.to_numeric(raw_df['rank'], errors='coerce')
    df_base['odds'] = pd.to_numeric(raw_df['odds'], errors='coerce').fillna(0)
    
    # Pre-filtering for simulation
    df_base = df_base[df_base['place_code'].notna()]
    unique_places = sorted(df_base['place_code'].unique().astype(str))

    # --- Power Loop ---
    for exponent in power_values:
        print(f"--- Simulating for Power: {exponent} ---")
        
        # Use a copy to calculate score
        df = df_base.copy()
        df['score'] = (df['win_prob'] ** exponent) * df['odds']
        
        summary_data = [] # For this exponent
        
        for score_thresh in min_scores:
            # Group by Place
            for p_code in unique_places:
                place_name = place_map.get(p_code, f"Place {p_code}")
                place_df = df[df['place_code'] == p_code]
                
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
        
        # Store summary for this power
        all_power_results[exponent] = pd.DataFrame(summary_data)

    # C. Generate Report
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
            .container {{ display: flex; flex-wrap: wrap; }}
            .box {{ margin-right: 20px; }}
        </style>
    </head>
    <body>
        <h1>Evaluation Report (Power Comparison)</h1>
        <p><strong>Period:</strong> {start_year} - {end_year}</p>
        <p><strong>Powers Tested:</strong> {power_values}</p>
        <p><strong>Generated:</strong> {pd.Timestamp.now()}</p>
    """
    
    # 1. Comparative Chart: ROI vs Score (For all powers)
    # Aggregating All Places
    html_content += "<h2>ROI Comparison by Power Exponent</h2>"
    
    plt.figure(figsize=(10, 6))
    
    best_configs = []
    
    for pow_val, res_df in all_power_results.items():
        # Aggregate across all places
        agg = res_df.groupby('min_score').agg({'bets': 'sum', 'cost': 'sum', 'return': 'sum'}).reset_index()
        agg['roi'] = (agg['return'] / agg['cost'] * 100).fillna(0)
        
        plt.plot(agg['min_score'], agg['roi'], marker='o', label=f'Power {pow_val}')
        
        # Find best ROI (with min bets > 10 to avoid noise?)
        valid_agg = agg[agg['bets'] >= 10]
        if not valid_agg.empty:
            best_row = valid_agg.loc[valid_agg['roi'].idxmax()]
            best_configs.append({
                'Power': pow_val,
                'Best ROI': best_row['roi'],
                'At Score': best_row['min_score'],
                'Bets': best_row['bets']
            })
            
    plt.axhline(100, color='red', linestyle='--', label='Break Even')
    plt.title("ROI vs Min Score Threshold (All Courses)")
    plt.xlabel("Min Score")
    plt.ylabel("ROI (%)")
    plt.grid(True)
    plt.legend()
    
    buf = BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    data_uri = base64.b64encode(buf.getvalue()).decode('utf-8')
    html_content += f'<div class="chart"><img src="data:image/png;base64,{data_uri}" style="max-width:100%"></div>'
    
    # 2. Best Configuration Table
    html_content += "<h2>Best Configuration Summary (Min 10 bets)</h2>"
    if best_configs:
        best_df = pd.DataFrame(best_configs).sort_values('Best ROI', ascending=False)
        html_content += best_df.to_html(classes='table', float_format="%.2f", index=False)
    else:
        html_content += "<p>No configurations with >10 bets found.</p>"

    # 3. Detailed Tables per Power
    for pow_val, res_df in all_power_results.items():
        html_content += f"<h2>Detailed Metrics (Power = {pow_val})</h2>"
        
        # Overall by score
        agg = res_df.groupby('min_score').agg({
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
        pivot_roi = res_df.pivot(index='min_score', columns='place_name', values='roi')
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
    parser.add_argument("--power_min", type=int, default=None, help="Min Power")
    parser.add_argument("--power_max", type=int, default=None, help="Max Power")
    parser.add_argument("--race_min", type=int, default=None, help="Min Race No")
    parser.add_argument("--race_max", type=int, default=None, help="Max Race No")
    args = parser.parse_args()
    
    generate_report(args.start, args.end, args.output, args.power_min, args.power_max, args.race_min, args.race_max)

