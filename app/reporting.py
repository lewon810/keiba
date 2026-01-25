import pandas as pd
import datetime
import os

def generate_html_report(predictions_dict, output_path="index.html"):
    """
    Generates an HTML report from a dictionary of predictions.
    predictions_dict: { "Title String": DataFrame, ... }
    """
    
    html_content = ["""
<!DOCTYPE html>
<html lang="ja">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Keiba AI Predictions</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body { background-color: #f8f9fa; }
        .race-card { margin-bottom: 20px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
        .score-high { color: #dc3545; font-weight: bold; }
        .header-section { margin-bottom: 30px; padding: 20px; background: white; border-radius: 8px; }
    </style>
</head>
<body>
    <div class="container py-4">
        <div class="header-section text-center">
            <h1>🏇 Keiba AI Predictions</h1>
            <p class="text-muted">Generated at: """ + datetime.datetime.now().strftime("%Y-%m-%d %H:%M") + """</p>
        </div>
"""]

    if not predictions_dict:
        html_content.append("""
        <div class="alert alert-warning" role="alert">
            No races found for the upcoming weekend.
        </div>
        """)
    else:
        # Sort keys to keep race order naturally if titles contain numbers, 
        # but dictionary order depends on insertion. Usually sorted by scraper.
        for title, df in predictions_dict.items():
            if df is None or df.empty:
                continue
                
            # Formatting for display
            display_df = df.copy()
            
            # Context extraction
            weather = display_df['weather'].iloc[0] if 'weather' in display_df.columns else 'Unknown'
            distance = display_df['distance'].iloc[0] if 'distance' in display_df.columns else 'Unknown'
            
            html_content.append(f"""
        <div class="card race-card">
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
            
            for i, (_, row) in enumerate(display_df.iterrows()):
                # Top 3 highlighting
                row_class = ""
                badge = ""
                if i == 0: 
                    row_class = "table-warning"
                    badge = "🥇 "
                elif i == 1: 
                    badge = "🥈 "
                elif i == 2: 
                    badge = "🥉 "
                
                odds = row.get('odds', '---.-')
                win_prob = row.get('win_prob', 0) * 100
                score = row.get('score', 0)
                
                html_content.append(f"""
                            <tr class="{row_class}">
                                <td>{badge}{i+1}</td>
                                <td>{row['name']}</td>
                                <td>{row['jockey']}</td>
                                <td>{odds}</td>
                                <td>{win_prob:.1f}%</td>
                                <td class="score-high">{score:.4f}</td>
                            </tr>
                """)
                
            html_content.append("""
                        </tbody>
                    </table>
                </div>
            </div>
        </div>
            """)

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
