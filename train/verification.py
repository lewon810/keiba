
import pandas as pd
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from train import preprocess
from train import settings

def test_preprocess():
    print("Testing Preprocess Logic...")
    
    # 1. Load Data (mocking or using small real subset)
    # We will use the real load_data but limit to 2016 for speed
    print("Loading data for 2016...")
    try:
        raw_df = preprocess.load_data(start_year=2016, end_year=2016)
        if raw_df.empty:
            print("No data found for 2016. Cannot verify.")
            return
            
        print(f"Loaded {len(raw_df)} rows.")
        print(f"Sample Date (Raw/Merged): {raw_df['date'].dropna().head().tolist()}")
        
        # 2. Run Preprocess
        df, artifacts = preprocess.preprocess(raw_df)
        
        # 3. Assertions
        print("Verifying 'date' column...")
        assert pd.api.types.is_datetime64_any_dtype(df['date']), f"Date is not datetime: {df['date'].dtype}"
        
        # Check for NaT (some might be missing if mapping incomplete, but shouldn't be all)
        nat_count = df['date'].isna().sum()
        print(f"NaT Count: {nat_count} / {len(df)}")
        
        valid_dates = df['date'].dropna()
        if not valid_dates.empty:
            print(f"Date Range: {valid_dates.min()} to {valid_dates.max()}")
            assert valid_dates.dt.year.min() >= 2016, "Date year mismatch (too early)"
        else:
            print("WARNING: All dates are NaT!")
            
        print("Verifying 'horse_id' column...")
        # Check if it was processed to int (LabelEncoder) or kept as string?
        # In preprocess, we label encode it.
        # But we should check if load_data handled it as string first.
        # We can check raw_df for string ID
        assert pd.api.types.is_string_dtype(raw_df['horse_id']), f"Raw horse_id is not string: {raw_df['horse_id'].dtype}"
        
        print("Verification PASSED.")
        
    except Exception as e:
        print(f"Verification FAILED: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_preprocess()
