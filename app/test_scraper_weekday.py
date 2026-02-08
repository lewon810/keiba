
import sys
import os
from datetime import datetime, timedelta

# Add project root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import scraper

def test_search():
    today = datetime.now()
    dates = [(today + timedelta(days=i)).strftime('%Y%m%d') for i in range(3)]
    
    print(f"Testing scraper for dates: {dates}")
    
    for d in dates:
        print(f"--- {d} ---")
        try:
            races = scraper.search_races(d)
            print(f"Found {len(races)} races.")
            for r in races[:3]:
                print(f"  {r['id']} {r['title']}")
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    test_search()
