import os
from datetime import datetime

def generate_index(output_file="public/index.html"):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Keiba AI Portal</title>
        <style>
            body {{ font-family: sans-serif; text-align: center; padding: 50px; }}
            a {{ display: inline-block; margin: 20px; padding: 20px; border: 1px solid #ccc; text-decoration: none; color: #333; border-radius: 8px; transition: background 0.3s; }}
            a:hover {{ background: #f0f0f0; }}
            h1 {{ color: #007bff; }}
        </style>
    </head>
    <body>
        <h1>Keiba AI Dashboard</h1>
        <p>Latest Update: {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>
        
        <div>
            <a href="predict.html">
                <h2>📈 Prediction</h2>
                <p>Weekend Race Predictions</p>
            </a>
            
            <a href="evaluate.html">
                <h2>📊 Evaluation</h2>
                <p>Model Performance & ROI Analysis</p>
            </a>
        </div>
    </body>
    </html>
    """
    
    with open(output_file, "w", encoding='utf-8') as f:
        f.write(html)
    print(f"Generated {output_file}")

if __name__ == "__main__":
    generate_index()
