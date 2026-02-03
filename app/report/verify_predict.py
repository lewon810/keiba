import sys
import os
import pandas as pd
from unittest.mock import MagicMock

# Add project root
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from app.report import predict_html_generator
from app import scraper, predictor

def verify_html_generation():
    print("Verifying HTML Generation with Mock Data...")
    
    # Mock Scraper
    scraper.search_races = MagicMock(return_value=[
        {'id': '202605010101', 'url': 'http://mock/1', 'title': 'Tokyo 1R Maiden', 'race_no': 1},
        {'id': '202605010102', 'url': 'http://mock/2', 'title': 'Tokyo 2R Maiden', 'race_no': 2},
        {'id': '202609020301', 'url': 'http://mock/3', 'title': 'Hanshin 1R Maiden', 'race_no': 1}
    ])
    
    scraper.fetch_race_data = MagicMock(return_value="dummy_html")
    
    # Mock Predictor
    # Return a DataFrame with minimal required columns
    mock_df = pd.DataFrame({
        'umaban': [1, 2, 3],
        'name': ['Horse A', 'Horse B', 'Horse C'],
        'jockey': ['Jockey A', 'Jockey B', 'Jockey C'],
        'odds': ['2.5', '3.0', '10.0'],
        'win_prob': [0.4, 0.3, 0.1],
        'weather': ['Sunny', 'Sunny', 'Sunny'],
        'distance': [1600, 1600, 1600],
        'course_type': ['Turf', 'Turf', 'Turf']
    })
    
    # Predictor returns DF directly
    predictor.predict = MagicMock(return_value=mock_df)
    
    # Run Generator
    output_file = "predict_test.html"
    predict_html_generator.generate_prediction_report(output_file)
    
    # Check if exists
    if os.path.exists(output_file):
        print("PASS: File generated.")
        
        with open(output_file, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # Check basic structure
        if "nav-tabs" in content and "Tokyo" in content and "Hanshin" in content:
            print("PASS: Tab structure found.")
        else:
            print("FAIL: Tab structure missing.")
            
        if "Power=4" in content:
             print("PASS: Correct Power displayed.")
        else:
             print("FAIL: Power not displayed.")
             
        if "🥇" in content and "🥈" in content:
            print("PASS: Medal emojis found.")
        else:
            print("FAIL: Medal emojis missing.")
             
    else:
        print("FAIL: File not generated.")

if __name__ == "__main__":
    verify_html_generation()
