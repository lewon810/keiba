import os
import sys
import pandas as pd
from datetime import datetime, timedelta

# Add project root
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from app import scraper, predictor

def generate_prediction_report(output_file="predict.html"):
    print("Generating Prediction Report...")
    
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
                df_pred = predictor.predict(race_data, return_df=True)
                
                if isinstance(df_pred, str): # Error message
                    print(df_pred)
                    continue
                    
                # Store
                all_results.append({
                    'date': date_str,
                    'race_id': race['id'],
                    'title': race['title'],
                    'df': df_pred
                })
            except Exception as e:
                print(f"Error processing {race['id']}: {e}")

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
    """
    
    if not all_results:
        html_content += "<p>No races found or prediction failed.</p>"
    else:
        # Sort by Date, then Time/ID?
        # race_id is usually chronological
        all_results.sort(key=lambda x: x['race_id'])
        
        for res in all_results:
            df = res['df']
            
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
                            <th>Score</th>
                        </tr>
                    </thead>
                    <tbody>
            """
            
            # Rows (Top 5?)
            for i, (_, row) in enumerate(df.head(10).iterrows()):
                symbol = ""
                if i == 0: symbol = "◎"
                elif i == 1: symbol = "○"
                elif i == 2: symbol = "▲"
                elif i == 3: symbol = "△"
                
                odds = row.get('odds', '---')
                win_prob = row.get('win_prob', 0) * 100
                score = row.get('score', 0)
                
                html_content += f"""
                        <tr>
                            <td><span class="symbol">{symbol}</span></td>
                            <td>{row.get('umaban', '')}</td>
                            <td>{row.get('name', '')}</td>
                            <td>{row.get('jockey', '')}</td>
                            <td>{odds}</td>
                            <td>{win_prob:.1f}%</td>
                            <td>{score:.4f}</td>
                        </tr>
                """
                
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
    # Ensure output dir
    os.makedirs("app/report", exist_ok=True)
    generate_prediction_report("predict.html")
