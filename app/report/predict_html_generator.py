import os
import sys
import pandas as pd
from datetime import datetime, timedelta
from collections import defaultdict

# Add project root
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from app import scraper, predictor, history_loader
from train import settings

SEX_MAP = {
    '牡': 'Male',
    '牝': 'Female',
    'セ': 'Gelding'
}

PLACE_MAP = {
    '01': 'Sapporo',
    '02': 'Hakodate',
    '03': 'Fukushima',
    '04': 'Niigata',
    '05': 'Tokyo',
    '06': 'Nakayama',
    '07': 'Chukyo',
    '08': 'Kyoto',
    '09': 'Hanshin',
    '10': 'Kokura'
}

def generate_prediction_report(output_file="predict.html", power_min=None, power_max=None):
    print("Generating Prediction Report (Tabbed View)...")
    
    # Load historical data to check availability
    history_loader.loader.load()
    has_historical_data = len(history_loader.loader.df) > 0
    print(f"Historical data available: {has_historical_data} ({len(history_loader.loader.df)} records)")
    
    # Defaults
    # User requested fixed default power 4, but let's keep args support just in case,
    # but strictly prioritize showing the single power score primarily.
    default_p = 4
    p_min = int(power_min) if power_min is not None else default_p
    p_max = int(power_max) if power_max is not None else p_min
    power_values = list(range(p_min, p_max + 1))
    
    # 1. Determine Target Dates (Today + 6 days)
    # The user requested to include all races for the week, not just weekends.
    today = datetime.now()
    target_dates = []
    
    for i in range(7):
        d = today + timedelta(days=i)
        target_dates.append(d.strftime('%Y%m%d'))

    print(f"Target Dates: {target_dates}")
    
    # 2. Search & Predict
    # Structure: data[date][venue_code] = list of race_results
    grouped_data = defaultdict(lambda: defaultdict(list))
    venue_names = {} # code -> name map for this run

    for date_str in target_dates:
        # Search all races
        races = scraper.search_races(date_str)
        print(f"Found {len(races)} races for {date_str}.")
        
        for race in races:
            try:
                # race['id'] = YYYYPP... (12 digits)
                # PP is 4:6
                race_id = race['id']
                place_code = race_id[4:6]
                venue_name = PLACE_MAP.get(place_code, f"Place {place_code}")
                venue_names[place_code] = venue_names.get(place_code, venue_name)
                
                print(f"Predicting {race_id} ({race['title']})...")
                # Scrape
                race_data = scraper.fetch_race_data(race['url'])
                if not race_data: continue
                
                # Predict
                df_pred = predictor.predict(race_data, return_df=True, power=p_min) # Start with min
                
                if isinstance(df_pred, str): # Error message
                    print(df_pred)
                    continue
                
                # Calculate scores
                def parse_odds(o):
                    try: return float(o)
                    except: return 0.0
                
                if 'odds_val' not in df_pred.columns:
                     df_pred['odds_val'] = df_pred['odds'].apply(parse_odds)
                     
                for p in power_values:
                    col_name = f'Score(P={p})'
                    df_pred[col_name] = (df_pred['win_prob'] ** p) * df_pred['odds_val']
                
                # Sort by default_p if present, else max
                sort_p = default_p if default_p in power_values else power_values[-1]
                df_pred = df_pred.sort_values(f'Score(P={sort_p})', ascending=False)
                
                # Meta
                weather = "?"
                dist = "?"
                course = "?"
                if not df_pred.empty and 'weather' in df_pred.columns:
                    weather = df_pred.iloc[0]['weather']
                    dist = df_pred.iloc[0]['distance']
                    course = df_pred.iloc[0]['course_type']

                grouped_data[date_str][place_code].append({
                    'id': race_id,
                    'title': race['title'],
                    'race_no': race['race_no'],
                    'df': df_pred,
                    'meta': f"{course} {dist}m {weather}"
                })
                
            except Exception as e:
                print(f"Error processing {race['id']}: {e}")
                import traceback
                traceback.print_exc()

    # 3. Generate HTML
    html_content = f"""
    <!DOCTYPE html>
    <html lang="ja">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1, shrink-to-fit=no">
        <title>Keiba Prediction Report</title>
        <!-- Bootstrap 4 CSS -->
        <link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css">
        <style>
            body {{ padding: 20px; background-color: #f4f7f6; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif; }}
            .card {{ margin-top: 20px; border: none; box-shadow: 0 2px 4px rgba(0,0,0,0.1); border-radius: 8px; overflow: hidden; }}
            
            /* Card Header: Black background, White text */
            .card-header {{ 
                background-color: #212529; 
                color: white; 
                border-bottom: none; 
                padding: 10px 15px;
            }}
            .race-title-text {{ font-size: 1.1em; font-weight: bold; }}
            .race-meta {{ font-size: 0.9em; color: #ced4da; float: right; }}
            
            /* Tabs */
            .nav-tabs {{ border-bottom: 2px solid #dee2e6; }}
            .nav-link {{ border: none; color: #495057; }}
            
            /* Level 1: Date (Standard Tabs) */
            #dateTabs .nav-link.active {{ 
                border-bottom: 3px solid #007bff; 
                color: #007bff; 
                font-weight: bold; 
                background: transparent;
            }}
            
            /* Level 2: Venue (Blue Pills/Tabs) */
            .venue-tabs .nav-link {{
                background-color: transparent;
                margin-right: 5px;
                border-radius: 4px;
                padding: 8px 15px;
            }}
            .venue-tabs .nav-link.active {{ 
                background-color: #007bff; 
                color: white !important; 
            }}
            
            /* Level 3: Race (Light Pills) */
            .race-tabs .nav-link {{
                padding: 5px 15px;
                margin-right: 5px;
                border: 1px solid transparent;
            }}
            .race-tabs .nav-link.active {{ 
                background-color: white; 
                color: #212529 !important; 
                font-weight: bold; 
                border: 1px solid #dee2e6;
                border-bottom: none;
                box-shadow: 0 -2px 5px rgba(0,0,0,0.05);
            }}
            
            .table th {{ background-color: #f8f9fa; border-top: none; font-weight: 600; }}
            .symbol {{ font-size: 1.2em; }}
            
            /* Top Pick: Yellow Highlight */
            .top-pick-row {{ background-color: #fff3cd !important; }}
            
            h1 {{ margin-bottom: 10px; text-align: center; color: #333; font-weight: 700; }}
            .header-icon {{ font-size: 1.5em; margin-right: 10px; }}
        </style>
    </head>
    <body>
        <div class="container-fluid" style="max-width: 800px; margin: 0 auto;">
            <h1><span class="header-icon">🏇</span>Keiba AI Predictions</h1>
            <p class="text-center text-muted" style="margin-bottom: 10px;">Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>
    """
    
    # Conditionally add warning banner if historical data is not available
    if not has_historical_data:
        html_content += """
            <!-- Warning Banner for Missing Historical Data -->
            <div class="alert alert-warning" role="alert" style="margin-bottom: 20px;">
                <strong>⚠️  Note:</strong> Historical race data is not available in the CI/CD environment. 
                Predictions are using default feature values, which may result in lower accuracy. 
                For better predictions, historical data should be provided in <code>train/data/raw</code>.
            </div>
        """
    else:
        html_content += f"""
            <!-- Historical Data Status -->
            <div class="alert alert-info" role="alert" style="margin-bottom: 20px;">
                <strong>✓ Info:</strong> Using historical race data ({len(history_loader.loader.df):,} records) for enhanced predictions.
            </div>
        """
    
    html_content += """
    """
    
    if not grouped_data:
        html_content += "<div class='alert alert-warning'>No race data found.</div>"
    else:
        # --- Level 1: Date Tabs ---
        html_content += '<ul class="nav nav-tabs" id="dateTabs" role="tablist">'
        sorted_dates = sorted(grouped_data.keys())
        for idx, d in enumerate(sorted_dates):
            active = "active" if idx == 0 else ""
            html_content += f"""
            <li class="nav-item">
                <a class="nav-link {active}" id="tab-{d}" data-toggle="tab" href="#content-{d}" role="tab">{d}</a>
            </li>
            """
        html_content += '</ul>'
        
        # Date Content
        html_content += '<div class="tab-content" id="dateTabsContent">'
        
        for idx, d in enumerate(sorted_dates):
            active = "show active" if idx == 0 else ""
            html_content += f'<div class="tab-pane fade {active}" id="content-{d}" role="tabpanel">'
            
            # --- Level 2: Venue Tabs ---
            venues = grouped_data[d]
            sorted_venues = sorted(venues.keys())
            
            html_content += f'<ul class="nav nav-tabs venue-tabs" id="venueTabs-{d}" role="tablist" style="margin-top: 10px;">'
            for v_idx, v_code in enumerate(sorted_venues):
                v_active = "active" if v_idx == 0 else ""
                v_name = venue_names.get(v_code, v_code)
                html_content += f"""
                <li class="nav-item">
                    <a class="nav-link {v_active}" id="tab-{d}-{v_code}" data-toggle="tab" href="#content-{d}-{v_code}" role="tab">{v_name}</a>
                </li>
                """
            html_content += '</ul>'
            
            # Venue Content
            html_content += f'<div class="tab-content" id="venueTabsContent-{d}">'
            for v_idx, v_code in enumerate(sorted_venues):
                v_active = "show active" if v_idx == 0 else ""
                races = sorted(venues[v_code], key=lambda x: x['race_no'])
                
                html_content += f'<div class="tab-pane fade {v_active}" id="content-{d}-{v_code}" role="tabpanel">'
                
                # --- Level 3: Race Tabs ---
                html_content += f'<ul class="nav nav-tabs race-tabs" id="raceTabs-{d}-{v_code}" role="tablist" style="margin-top: 10px;">'
                for r_idx, race in enumerate(races):
                    r_active = "active" if r_idx == 0 else ""
                    rid = race['id']
                    html_content += f"""
                    <li class="nav-item">
                        <a class="nav-link {r_active}" id="tab-{rid}" data-toggle="tab" href="#content-{rid}" role="tab">{race['race_no']}R</a>
                    </li>
                    """
                html_content += '</ul>'
                
                # Race Content
                html_content += f'<div class="tab-content" id="raceTabsContent-{d}-{v_code}">'
                
                for r_idx, race in enumerate(races):
                    r_active = "show active" if r_idx == 0 else ""
                    rid = race['id']
                    df = race['df']
                    meta = race['meta']
                    
                    html_content += f"""
                    <div class="tab-pane fade {r_active}" id="content-{rid}" role="tabpanel">
                        <div class="card">
                            <div class="card-header">
                                <span class="race-title-text">{race['title']}</span>
                                <span class="race-meta">{meta} | ID: {rid}</span>
                            </div>
                            <div class="card-body p-0">
                                <table class="table table-hover table-sm mb-0">
                                    <thead>
                                        <tr>
                                            <th>Mark</th>
                                            <th>#</th>
                                            <th>Horse</th>
                                            <th>Jockey</th>
                                            <th>Odds</th>
                                            <th>Win%</th>
                                            <th>Score (P={default_p})</th>
                                        </tr>
                                    </thead>
                                    <tbody>
                    """
                    
                    for i, (_, row) in enumerate(df.head(12).iterrows()): # Show top 12
                        symbol = ""
                        row_class = ""
                        if i == 0: 
                            symbol = "🥇" # Gold Medal
                            row_class = "top-pick-row"
                        elif i == 1: symbol = "🥈" # Silver
                        elif i == 2: symbol = "🥉" # Bronze
                        elif i == 3: symbol = "△"
                        elif i == 4: symbol = "☆"
                        
                        odds = row.get('odds', '---')
                        win_prob = row.get('win_prob', 0) * 100
                        score = row.get(f'Score(P={default_p})', 0)
                        
                        html_content += f"""
                                        <tr class="{row_class}">
                                            <td><span class="symbol">{symbol}</span></td>
                                            <td>{row.get('umaban', '')}</td>
                                            <td>{row.get('name', '')}</td>
                                            <td>{row.get('jockey', '')}</td>
                                            <td>{odds}</td>
                                            <td>{win_prob:.1f}%</td>
                                            <td><strong>{score:.4f}</strong></td>
                                        </tr>
                        """
                        
                    html_content += """
                                    </tbody>
                                </table>
                            </div>
                        </div>
                    </div>
                    """
                
                html_content += '</div>' # End Race Content
                html_content += '</div>' # End Venue Pane
            
            html_content += '</div>' # End Venue Tabs Content
            html_content += '</div>' # End Date Pane
            
        html_content += '</div>' # End Date Tabs Content

    html_content += """
        </div>
        <!-- Bootstrap JS and dependencies -->
        <script src="https://code.jquery.com/jquery-3.5.1.slim.min.js"></script>
        <script src="https://cdn.jsdelivr.net/npm/@popperjs/core@2.5.4/dist/umd/popper.min.js"></script>
        <script src="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/js/bootstrap.min.js"></script>
    </body>
    </html>
    """
    
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
