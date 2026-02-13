import pandas as pd
import os
import glob
from train import settings

class HistoryLoader:
    def __init__(self):
        self.df = None
        self.is_loaded = False
        
    def load(self):
        if self.is_loaded: return
        
        print("Loading historical data for inference...")
        # Load all CSVs in raw data
        # In a real app, this might be a database or a optimized parquet file
        files = glob.glob(os.path.join(settings.RAW_DATA_DIR, "*.csv"))
        dfs = []
        for f in files:
            try:
                # Optimized load: only needed columns
                # We need date to sort, rank/speed_index for metrics, horse_id for key
                # speed_index is computed in preprocess, not in raw!
                # Raw data only has Time. We need to re-calc SI or use 'time' as proxy if SI not saved suitable.
                # Actually, `learn/preprocess.py` calculates SI.
                # If we want accurate Lag SI, we need to load *Processed* data or re-process raw history.
                # For Phase 4 MVP, let's load Raw, and just use Rank and Time as Lag features.
                # Or Recalculate SI on the fly (expensive).
                # BETTER: Save a "processed_history.parquet" during training!
                # But we didn't do that.
                # Plan B: Load Raw, Parse Date, Sort. Lag Rank is easy. Lag SI is hard without course stats.
                # We will just use Rank for now as proof of concept.
                
                df = pd.read_csv(f, usecols=['horse_id', 'date', 'rank', 'time', 'race_id', 'last_3f'])
                dfs.append(df)
            except Exception as e:
                # Columns might be missing if mixing old/new csvs
                pass
                
        if dfs:
            self.df = pd.concat(dfs, ignore_index=True)
            # Parse Date
            self.df['date'] = pd.to_datetime(self.df['date'], format='%Y年%m月%d日', errors='coerce')
            self.df = self.df.dropna(subset=['date'])
            # Parse last_3f to numeric
            self.df['last_3f'] = pd.to_numeric(self.df['last_3f'], errors='coerce').fillna(0)
            self.df = self.df.sort_values('date') 
        else:
            self.df = pd.DataFrame(columns=['horse_id', 'date', 'rank'])
            
        self.is_loaded = True
        print(f"History loaded: {len(self.df)} records.")

    def get_last_race(self, horse_id, current_date_str=None):
        """
        Returns dict of last race stats: {lag1_rank, interval, ...}
        """
        if self.df is None or self.df.empty:
             return None
             
        # Filter by horse
        # horse_id in scraper is string
        history = self.df[self.df['horse_id'].astype(str) == str(horse_id)]
        
        if history.empty:
            return None
            
        # Filter before current date if provided
        if current_date_str:
             curr_date = pd.to_datetime(current_date_str)
             history = history[history['date'] < curr_date]
             
        if history.empty:
            return None
            
        last_race = history.iloc[-1]
        
        # Calculate Interval
        interval = 365
        if current_date_str:
            curr_date = pd.to_datetime(current_date_str)
            interval = (curr_date - last_race['date']).days
            
        # Parse Rank
        try:
            rank = int(last_race['rank'])
        except:
            rank = 99
            
        # Parse last_3f
        try:
            last_3f = float(last_race.get('last_3f', 0))
        except:
            last_3f = 0.0
            
        return {
            "lag1_rank": rank,
            "interval": interval,
            "lag1_speed_index": 0, # Placeholder as we don't have stored SI in raw
            "lag1_last_3f": last_3f
        }

# Global instance
loader = HistoryLoader()
