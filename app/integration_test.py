import scraper
import predictor
import sys
import os

# Ensure imports work if running from root or src
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_integration():
    url = "https://race.netkeiba.com/race/shutuba.html?race_id=202606010809&rf=race_list"
    print(f"Testing with URL: {url}")
    
    try:
        data = scraper.fetch_race_data(url)
        if not data:
            print("Scraper returned empty data!")
            return
            
        print(f"Scraper returned {len(data)} items.")
        
        result = predictor.predict(data)
        print("Prediction Result:")
        print(result)
        print("Integration Test Passed!")
        
    except Exception as e:
        print(f"Integration Test Failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_integration()
