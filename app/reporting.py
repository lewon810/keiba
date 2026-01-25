import pandas as pd
import datetime
import os

PLACE_MAP = {
    "01": "札幌 (Sapporo)", "02": "函館 (Hakodate)", "03": "福島 (Fukushima)", "04": "新潟 (Niigata)",
    "05": "東京 (Tokyo)", "06": "中山 (Nakayama)", "07": "中京 (Chukyo)", "08": "京都 (Kyoto)",
    "09": "阪神 (Hanshin)", "10": "小倉 (Kokura)"
}

def generate_html_report(predictions_list, output_path="index.html"):
    """
    Generates a Tabbed HTML report from a list of prediction dicts.
    predictions_list: [ { "date":..., "place":..., "race_no":..., "df":... }, ... ]
    """
    
    # 1. Structure Data: Date -> Place -> Races
    data_tree = {}
    for item in predictions_list:
        d = item['date']
        p = item['place']
        if d not in data_tree: data_tree[d] = {}
        if p not in data_tree[d]: data_tree[d][p] = []
        data_tree[d][p].append(item)
    
    # Sort keys
    sorted_dates = sorted(data_tree.keys())
    
    html_content = ["""
<!DOCTYPE html>
<html lang="ja">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Keiba AI Predictions</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
    <style>
        body { background-color: #f8f9fa; }
        .race-card { margin-bottom: 20px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
        .score-high { color: #dc3545; font-weight: bold; }
        .header-section { margin-bottom: 20px; padding: 20px; background: white; border-radius: 8px; }
        .nav-pills .nav-link.active { background-color: #0d6efd; }
        .race-btn-group { margin-bottom: 15px; flex-wrap: wrap; }
    </style>
</head>
<body>
    <div class="container py-4">
        <div class="header-section text-center">
            <h1>🏇 Keiba AI Predictions</h1>
            <p class="text-muted">Generated at: """ + datetime.datetime.now().strftime("%Y-%m-%d %H:%M") + """</p>
        </div>
"""]

    if not predictions_list:
        html_content.append("""
        <div class="alert alert-warning" role="alert">
            No races found.
        </div>
        """)
    else:
        # --- Level 1: Date Tabs ---
        html_content.append('<ul class="nav nav-tabs mb-3" id="dateTabs" role="tablist">')
        for i, date_str in enumerate(sorted_dates):
            active = "active" if i == 0 else ""
            html_content.append(f"""
                <li class="nav-item" role="presentation">
                    <button class="nav-link {active}" id="tab-{date_str}" data-bs-toggle="tab" data-bs-target="#content-{date_str}" type="button" role="tab">{date_str}</button>
                </li>
            """)
        html_content.append('</ul>')
        
        # --- Level 1 Content ---
        html_content.append('<div class="tab-content" id="dateTabsContent">')
        
        for i, date_str in enumerate(sorted_dates):
            active = "show active" if i == 0 else ""
            html_content.append(f'<div class="tab-pane fade {active}" id="content-{date_str}" role="tabpanel">')
            
            places = sorted(data_tree[date_str].keys())
            
            # --- Level 2: Place Pills ---
            html_content.append(f'<ul class="nav nav-pills mb-3" id="placeTabs-{date_str}" role="tablist">')
            for j, p_code in enumerate(places):
                p_name = PLACE_MAP.get(p_code, f"Place {p_code}")
                active_p = "active" if j == 0 else ""
                html_content.append(f"""
                    <li class="nav-item" role="presentation">
                        <button class="nav-link {active_p}" id="tab-{date_str}-{p_code}" data-bs-toggle="pill" data-bs-target="#content-{date_str}-{p_code}" type="button" role="tab">{p_name}</button>
                    </li>
                """)
            html_content.append('</ul>')
            
            # --- Level 2 Content ---
            html_content.append(f'<div class="tab-content" id="placeTabsContent-{date_str}">')
            
            for j, p_code in enumerate(places):
                active_p = "show active" if j == 0 else ""
                html_content.append(f'<div class="tab-pane fade {active_p}" id="content-{date_str}-{p_code}" role="tabpanel">')
                
                # Races in this place
                races = sorted(data_tree[date_str][p_code], key=lambda x: x['race_no'])
                
                # --- Level 3: Race Buttons (Anchor Links to cards) ---
                # Actually, displaying 12 cards vertically is fine if we have Place tabs.
                # Let's add a quick jump list though.
                html_content.append('<div class="btn-group race-btn-group" role="group">')
                for item in races:
                    r_no = item['race_no']
                    html_content.append(f'<a href="#race-{date_str}-{p_code}-{r_no}" class="btn btn-outline-secondary">{r_no}R</a>')
                html_content.append('</div>')

                # Render Race Cards
                for item in races:
                    r_no = item['race_no']
                    df = item['df']
                    title = item['title']
                    
                    # Formatting matching previous style
                    weather = df['weather'].iloc[0] if 'weather' in df.columns else 'Unknown'
                    distance = df['distance'].iloc[0] if 'distance' in df.columns else 'Unknown'

                    html_content.append(f"""
                    <div class="card race-card" id="race-{date_str}-{p_code}-{r_no}">
                        <div class="card-header bg-dark text-white d-flex justify-content-between align-items-center">
                            <h5 class="mb-0">{title}</h5>
                            <span class="badge bg-secondary">{weather} / {distance}m</span>
                        </div>
                        <div class="card-body p-0">
                            <div class="table-responsive">
                                <table class="table table-hover mb-0">
                                    <thead class="table-light">
                                        <tr>
                                            <th>#</th>
                                            <th>Name</th>
                                            <th>Jockey</th>
                                            <th>Odds</th>
                                            <th>Win%</th>
                                            <th>Score</th>
                                        </tr>
                                    </thead>
                                    <tbody>
                    """)
                    
                    for k, (_, row) in enumerate(df.iterrows()):
                        row_class = ""
                        badge = ""
                        if k == 0: 
                            row_class = "table-warning"
                            badge = "🥇 "
                        elif k == 1: badge = "🥈 "
                        elif k == 2: badge = "🥉 "
                        
                        odds = row.get('odds', '---.-')
                        win_prob = row.get('win_prob', 0) * 100
                        score = row.get('score', 0)
                        
                        html_content.append(f"""
                                    <tr class="{row_class}">
                                        <td>{badge}{k+1}</td>
                                        <td>{row['name']}</td>
                                        <td>{row['jockey']}</td>
                                        <td>{odds}</td>
                                        <td>{win_prob:.1f}%</td>
                                        <td class="score-high">{score:.4f}</td>
                                    </tr>
                        """)
                    html_content.append("</tbody></table></div></div></div>")
                
                html_content.append('</div>') # End Place Content
            
            html_content.append('</div>') # End Place Tabs
            html_content.append('</div>') # End Date Content

        html_content.append('</div>') # End Date Tabs

    html_content.append("""
        <footer class="text-center text-muted mt-5">
            <small>Powered by Keiba AI | LightGBM Ranker</small>
        </footer>
    </div>
</body>
</html>
""")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(html_content))
    
    print(f"Report generated: {output_path}")
