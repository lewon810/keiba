import numpy as np
from train.evaluate import evaluate
import pandas as pd
import argparse

def grid_search_roi(start, end, csv):
    scores = np.arange(0.0, 3.1, 0.1) # 0.0 to 3.0
    results = []

    print(f"Starting Grid Search for Min ROI Score (0.0 - 3.0) on {start}-{end}...")
    
    for score in scores:
        score = round(score, 1)
        print(f"\n>> Testing min_score = {score}")
        
        # Suppress stdout to keep it clean if possible, but for now we let it print
        metrics = evaluate(start, end, csv_file=csv, min_score=score)
        
        if metrics:
            metrics['min_score'] = score
            results.append(metrics)
    
    if not results:
        print("No results collected.")
        return

    # Create Summary DataFrame
    df_results = pd.DataFrame(results)
    
    # Select columns to show
    cols = ['min_score', 'bet_races', 'hit_rate', 'roi', 'total_return', 'total_bet']
    summary = df_results[cols]
    
    print("\n\n====== Grid Search Results ======")
    print(summary.to_string(index=False))
    
    # Optional: Find best ROI
    if not summary.empty and summary['bet_races'].sum() > 0:
        # Filter where we bet at least 10 races to avoid noise (optional)
        valid_summary = summary[summary['bet_races'] > 0]
        if not valid_summary.empty:
            best_run = valid_summary.loc[valid_summary['roi'].idxmax()]
            print("\nSuggesting Best Threshold based on ROI:")
            print(best_run)
        else:
            print("\nNo bets were made in any threshold.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, default=2025)
    parser.add_argument("--end", type=int, default=2025)
    parser.add_argument("--csv", type=str, help="Path to existing CSV file")
    args = parser.parse_args()
    
    grid_search_roi(args.start, args.end, csv=args.csv)
