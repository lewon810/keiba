import pandas as pd
import numpy as np
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
                
                # 必要カラム（speed_index計算のため course_type, distance, time を追加）
                needed = ['horse_id', 'rank', 'time', 'race_id', 'last_3f',
                          'course_type', 'distance']
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
            
            # --- Speed Index の計算 ---
            self._calculate_speed_index()
 
        else:
            self.df = pd.DataFrame(columns=['horse_id', 'date', 'rank'])
            
        self.is_loaded = True
        print(f"History loaded: {len(self.df)} records.")
        
        # Warning if no historical data
        if len(self.df) == 0:
            print("⚠️  WARNING: No historical data found!")
            print("⚠️  Predictions will use default values, resulting in lower accuracy.")
            print(f"⚠️  Expected data location: {settings.RAW_DATA_DIR}")

    def _calculate_speed_index(self):
        """
        履歴データから speed_index を計算する。
        preprocess.py と同じロジック: Z-score (コース種別 × 距離別)
        speed_index = (course_mean - time_sec) / course_std
        （高い値 = 速い）
        """
        # time を秒数にパース
        def parse_time(t_str):
            try:
                t_str = str(t_str)
                if ':' in t_str:
                    m, s = t_str.split(':')
                    return int(m) * 60 + float(s)
                return float(t_str)
            except:
                return np.nan
        
        self.df['time_sec'] = self.df['time'].apply(parse_time)
        
        # コース × 距離 ごとの統計を計算
        if 'course_type' in self.df.columns and 'distance' in self.df.columns:
            self.df['distance'] = pd.to_numeric(self.df['distance'], errors='coerce')
            valid_times = self.df[self.df['time_sec'] > 0]
            
            if not valid_times.empty:
                course_stats = valid_times.groupby(
                    ['course_type', 'distance']
                )['time_sec'].agg(['mean', 'std']).reset_index()
                course_stats.columns = ['course_type', 'distance', 'course_mean', 'course_std']
                
                # マージして speed_index を計算
                self.df = self.df.merge(course_stats, on=['course_type', 'distance'], how='left')
                self.df['speed_index'] = (
                    (self.df['course_mean'] - self.df['time_sec']) / 
                    self.df['course_std'].replace(0, 1)
                )
                self.df['speed_index'] = self.df['speed_index'].fillna(0)
                
                # 一時カラムの削除
                self.df = self.df.drop(columns=['course_mean', 'course_std'], errors='ignore')
                
                print(f"Speed index calculated: mean={self.df['speed_index'].mean():.3f}, "
                      f"std={self.df['speed_index'].std():.3f}")
            else:
                self.df['speed_index'] = 0
        else:
            self.df['speed_index'] = 0

    def get_last_race(self, horse_id, current_date_str=None):
        """
        Returns dict of last race stats: {lag1_rank, interval, lag1_speed_index, lag1_last_3f}
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
        
        # Speed Index（計算済みの値を使用）
        try:
            speed_index = float(last_race.get('speed_index', 0))
        except:
            speed_index = 0.0
            
        return {
            "lag1_rank": rank,
            "interval": interval,
            "lag1_speed_index": speed_index,
            "lag1_last_3f": last_3f
        }

# Global instance
loader = HistoryLoader()
