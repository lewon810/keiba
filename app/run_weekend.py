import datetime
from . import scraper
from . import predictor
from . import reporting
import sys

def get_weekend_dates():
    """
    Returns a list of date strings (YYYYMMDD) for the upcoming/current weekend (Sat, Sun).
    If today is Monday-Friday, returns the *next* weekend.
    If today is Saturday or Sunday, returns *this* weekend (Sat+Sun).
    """
    today = datetime.date.today()
    weekday = today.weekday() # Mon=0, Sun=6
    
    # Calculate days to Saturday
    # If today is Sat (5), 0 days. If Sun (6), -1 days (so Sat was yesterday).
    # If today is Mon (0), 5 days.
    
    if weekday == 6: # Sunday
        saturday = today - datetime.timedelta(days=1)
    else:
        days_ahead = 5 - weekday
        if days_ahead < 0: # Should not happen if handled Sat/Sun above, but for safety
             days_ahead += 7
        saturday = today + datetime.timedelta(days=days_ahead)
        
    sunday = saturday + datetime.timedelta(days=1)
    
    return [
        saturday.strftime("%Y%m%d"),
        sunday.strftime("%Y%m%d")
    ]

def main():
    dates = get_weekend_dates()
    print(f"Targeting Weekend: {dates}")
    
    all_predictions = []
    
    for date_str in dates:
        print(f"\nSearching races for {date_str}...")
        try:
            races = scraper.search_races(date_str)
            if not races:
                print(f"No races found for {date_str}.")
                continue
                
            print(f"Found {len(races)} races.")
            for r in races:
                title = f"{date_str} {r['title']}"
                print(f"Predicting: {title}")
                
                race_data = scraper.fetch_race_data(r['url'])
                if race_data:
                    # Use return_df=True to get DataFrame
                    df = predictor.predict(race_data, return_df=True)
                    if isinstance(df, str): # Error message
                        print(f"Prediction failed: {df}")
                    else:
                        # r dict contains: 'id', 'url', 'title', 'race_no'
                        # id is like 202606010802 -> YYYYPP...
                        # Extract Place Code
                        rid = r['id']
                        p_code = rid[4:6]
                        
                        all_predictions.append({
                            "date": date_str,
                            "place": p_code,
                            "race_no": r['race_no'],
                            "title": title,
                            "df": df
                        })
                        
        except Exception as e:
            print(f"Error processing {date_str}: {e}")
            import traceback
            traceback.print_exc()

    # Generate Report
    print(f"\nGenerating Report for {len(all_predictions)} races...")
    import os
    os.makedirs("public", exist_ok=True)
    reporting.generate_html_report(all_predictions, output_path="public/index.html")

if __name__ == "__main__":
    main()
