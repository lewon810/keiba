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
        files = glob.glob(os.path.join(settings.RAW_DATA_DIR, "*.csv"))
        dfs = []
        for f in files:
            try:
                # results_*.csv のみ対象
                if not os.path.basename(f).startswith('results_'):
                    continue
                    
                # 新フォーマット: year, month, day カラム
                # レガシー: date カラム
                df = pd.read_csv(f, dtype={'race_id': str, 'horse_id': str})
                
                # 必要カラムのみ残す
                needed = ['horse_id', 'rank', 'time', 'race_id', 'last_3f']
                date_cols = []
                if 'year' in df.columns and 'month' in df.columns and 'day' in df.columns:
                    date_cols = ['year', 'month', 'day']
                elif 'date' in df.columns:
                    date_cols = ['date']
                
                keep_cols = [c for c in needed + date_cols if c in df.columns]
                df = df[keep_cols]
                dfs.append(df)
            except Exception as e:
                pass
                
        if dfs:
            self.df = pd.concat(dfs, ignore_index=True)
            
            # 日付の構築
            if 'year' in self.df.columns and 'month' in self.df.columns and 'day' in self.df.columns:
                # 新フォーマット: year, month, day から datetime を構築
                self.df['date'] = pd.to_datetime(self.df[['year', 'month', 'day']], errors='coerce')
                self.df = self.df.dropna(subset=['date'])
                self.df = self.df.sort_values('date')
            elif 'date' in self.df.columns:
                # レガシーフォールバック
                if self.df['date'].dtype == 'int64' or self.df['date'].dtype == 'int32':
                    print("⚠️  Warning: Date column is integer type. Sorting by race_id.")
                    self.df = self.df.sort_values('race_id')
                    def extract_date_from_race_id(rid):
                        try:
                            rid_str = str(rid)
                            if len(rid_str) >= 12:
                                year = rid_str[0:4]
                                month = rid_str[6:8]
                                day = rid_str[8:10]
                                return pd.to_datetime(f"{year}-{month}-{day}", errors='coerce')
                            return pd.NaT
                        except:
                            return pd.NaT
                    self.df['date'] = self.df['race_id'].apply(extract_date_from_race_id)
                    self.df = self.df.dropna(subset=['date'])
                else:
                    self.df['date'] = pd.to_datetime(self.df['date'], format='%Y年%m月%d日', errors='coerce')
                    self.df = self.df.dropna(subset=['date'])
                    self.df = self.df.sort_values('date')
            else:
                self.df = self.df.sort_values('race_id')
                self.df['date'] = pd.NaT
            
            # Parse last_3f to numeric
            self.df['last_3f'] = pd.to_numeric(self.df['last_3f'], errors='coerce').fillna(0)
 
        else:
            self.df = pd.DataFrame(columns=['horse_id', 'date', 'rank'])
            
        self.is_loaded = True
        print(f"History loaded: {len(self.df)} records.")
        
        # Warning if no historical data
        if len(self.df) == 0:
            print("⚠️  WARNING: No historical data found!")
            print("⚠️  Predictions will use default values, resulting in lower accuracy.")
            print(f"⚠️  Expected data location: {settings.RAW_DATA_DIR}")

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
