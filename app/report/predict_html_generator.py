import os
import sys
import pandas as pd
from datetime import datetime, timedelta

# Add project root
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from app import scraper, predictor
from train import settings

def generate_prediction_report(output_file="predict.html", power_min=None, power_max=None):
    print("Generating Prediction Report...")
    
    # Defaults
    p_min = int(power_min) if power_min is not None else settings.POWER_EXPONENT
    p_max = int(power_max) if power_max is not None else p_min
    power_values = list(range(p_min, p_max + 1))
    print(f"Prediction Power Exponents: {power_values}")

    # 1. Determine Target Date (Saturday or Sunday)
    today = datetime.now()
    target_dates = []
    
    # If today is Sat/Sun, include today. If Fri, include Sat/Sun.
    if today.weekday() == 4: # Friday
        target_dates.append((today + timedelta(days=1)).strftime('%Y%m%d'))
        target_dates.append((today + timedelta(days=2)).strftime('%Y%m%d'))
    elif today.weekday() == 5: # Saturday
        target_dates.append(today.strftime('%Y%m%d'))
        target_dates.append((today + timedelta(days=1)).strftime('%Y%m%d'))
    elif today.weekday() == 6: # Sunday
        target_dates.append(today.strftime('%Y%m%d'))
    else:
        # Debug/Dev: Force next Saturday for testing if weekday
        print("Not a weekend. Searching for next Saturday...")
        days_ahead = 5 - today.weekday()
        if days_ahead <= 0: days_ahead += 7
        target_dates.append((today + timedelta(days=days_ahead)).strftime('%Y%m%d'))

    print(f"Target Dates: {target_dates}")
    
    # 2. Search & Predict
    all_results = []
    
    for date_str in target_dates:
        # Search all races
        races = scraper.search_races(date_str)
        print(f"Found {len(races)} races for {date_str}.")
        
        for race in races:
            try:
                print(f"Predicting {race['id']} ({race['title']})...")
                # Scrape
                race_data = scraper.fetch_race_data(race['url'])
                if not race_data: continue
                
                # Predict
                # Call with p_min just to get base DF
                df_pred = predictor.predict(race_data, return_df=True, power=p_min)
                
                if isinstance(df_pred, str): # Error message
                    print(df_pred)
                    continue
                
                # Calculate scores for ALL powers
                # Ensure odds are numeric
                def parse_odds(o):
                    try: return float(o)
                    except: return 0.0
                    
                # Odds usually come as strings in 'odds' column or parsed? 
                # predictor returns 'odds_val' in df usually
                if 'odds_val' not in df_pred.columns:
                     df_pred['odds_val'] = df_pred['odds'].apply(parse_odds)
                     
                for p in power_values:
                    col_name = f'Score(P={p})'
                    df_pred[col_name] = (df_pred['win_prob'] ** p) * df_pred['odds_val']
                    
                # Sort by the LAST power (highest exponent -> favors winners+odds most aggressively?)
                # Or just keep default sort from predictor?
                # Predictor sorts by 'score' (which was p_min).
                # Let's sort by max power or default 4
                sort_p = 4 if 4 in power_values else power_values[-1]
                df_pred = df_pred.sort_values(f'Score(P={sort_p})', ascending=False)
                    
                # Store
                all_results.append({
                    'date': date_str,
                    'race_id': race['id'],
                    'title': race['title'],
                    'df': df_pred,
                    'sort_power': sort_p
                })
            except Exception as e:
                print(f"Error processing {race['id']}: {e}")
                import traceback
                traceback.print_exc()

    # 3. Generate HTML
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Keiba Prediction</title>
        <style>
            body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 0; padding: 20px; background-color: #f9f9f9; }}
            h1 {{ color: #333; text-align: center; }}
            .race-container {{ background: white; border-radius: 8px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); margin-bottom: 20px; padding: 15px; }}
            .race-header {{ border-bottom: 2px solid #007bff; padding-bottom: 10px; margin-bottom: 10px; display: flex; justify-content: space-between; align-items: center; }}
            .race-title {{ font-size: 1.2em; font-weight: bold; color: #007bff; }}
            .race-meta {{ color: #666; font-size: 0.9em; }}
            table {{ border-collapse: collapse; width: 100%; }}
            th, td {{ padding: 10px; text-align: left; border-bottom: 1px solid #eee; }}
            th {{ background-color: #f8f9fa; color: #555; }}
            tr:first-child td {{ font-weight: bold; background-color: #fff3cd; }} /* Top Pick Highlight */
            .symbol {{ font-weight: bold; width: 30px; display: inline-block; text-align: center; }}
            .score-bar {{ height: 4px; background-color: #28a745; }}
        </style>
    </head>
    <body>
        <h1>Weekend Predictions</h1>
        <p style="text-align:center;">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>
        <p style="text-align:center;">Powers: {power_values}</p>
    """
    
    if not all_results:
        html_content += "<p>No races found or prediction failed.</p>"
    else:
        # Sort by Date, then Time/ID
        all_results.sort(key=lambda x: x['race_id'])
        
        for res in all_results:
            df = res['df']
            sort_p = res['sort_power']
            
            # Context
            weather = "?"
            dist = "?"
            if not df.empty and 'weather' in df.columns:
                weather = df.iloc[0]['weather']
                dist = df.iloc[0]['distance']
            
            html_content += f"""
            <div class="race-container">
                <div class="race-header">
                    <span class="race-title">{res['title']}</span>
                    <span class="race-meta">ID: {res['race_id']} | {weather} | {dist}m</span>
                </div>
                <table>
                    <thead>
                        <tr>
                            <th>Mark</th>
                            <th>#</th>
                            <th>Horse</th>
                            <th>Jockey</th>
                            <th>Odds</th>
                            <th>Win%</th>
            """
            
            # Dynamic Score Headers
            for p in power_values:
                 html_content += f"<th>Score(P={p})</th>"
                 
            html_content += """
                        </tr>
                    </thead>
                    <tbody>
            """
            
            # Rows (Top 10)
            for i, (_, row) in enumerate(df.head(10).iterrows()):
                symbol = ""
                if i == 0: symbol = "◎"
                elif i == 1: symbol = "○"
                elif i == 2: symbol = "▲"
                elif i == 3: symbol = "△"
                
                odds = row.get('odds', '---')
                win_prob = row.get('win_prob', 0) * 100
                
                html_content += f"""
                        <tr>
                            <td><span class="symbol">{symbol}</span></td>
                            <td>{row.get('umaban', '')}</td>
                            <td>{row.get('name', '')}</td>
                            <td>{row.get('jockey', '')}</td>
                            <td>{odds}</td>
                            <td>{win_prob:.1f}%</td>
                """
                
                for p in power_values:
                    score = row.get(f'Score(P={p})', 0)
                    style = "font-weight:bold" if p == sort_p else ""
                    html_content += f'<td style="{style}">{score:.4f}</td>'
                    
                html_content += "</tr>"
                
            html_content += """
                    </tbody>
                </table>
            </div>
            """

    html_content += "</body></html>"
    
    with open(output_file, "w", encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"Saved {output_file}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--power_min", type=int, default=None, help="Min Power")
    parser.add_argument("--power_max", type=int, default=None, help="Max Power")
    args = parser.parse_args()

    # Ensure output dir
    os.makedirs("app/report", exist_ok=True)
    generate_prediction_report("predict.html", power_min=args.power_min, power_max=args.power_max)
