
import pandas as pd
import os
import settings

def verify_columns():
    print("results_*.csvファイルのカラム順序を検証しています...")
    
    desired_order = [
         "race_id", "course_type", "distance", "weather", "condition",
         "year", "month", "day",
         "rank", "waku", "umaban", "horse_name", "horse_id", 
         "jockey", "jockey_id", "trainer", "trainer_id", 
         "horse_weight", "weight_diff", "time", 
         "passing", "last_3f", "odds", "popularity"
    ]
    
    files = [f for f in os.listdir(settings.RAW_DATA_DIR) if f.startswith('results_') and f.endswith('.csv')]
    
    for f in files:
        path = os.path.join(settings.RAW_DATA_DIR, f)
        df = pd.read_csv(path, nrows=1)
        cols = df.columns.tolist()
        
        # 最初のN列が期待通りか確認
        match_count = 0
        mean_match = True
        
        for i, col in enumerate(desired_order):
            if i < len(cols):
                if cols[i] != col:
                    if col in ('race_id', 'year', 'month', 'day'):
                        print(f"FAIL {f}: インデックス {i} に '{col}' が期待されましたが、{cols[i]} が見つかりました")
                        mean_match = False
        
        if mean_match:
            # year, month, day カラムの整数チェック
            try:
                sample = pd.read_csv(path, nrows=5)
                if 'year' in sample.columns and 'month' in sample.columns and 'day' in sample.columns:
                    year_val = sample['year'].iloc[0]
                    month_val = sample['month'].iloc[0]
                    day_val = sample['day'].iloc[0]
                    if isinstance(year_val, (int, float)) and 2000 <= year_val <= 2030:
                        print(f"PASS {f}: 順序OK, 日付OK (year={int(year_val)}, month={int(month_val)}, day={int(day_val)})")
                    else:
                        print(f"WARN {f}: 順序OKですが、year値が不正です (year={year_val})")
                else:
                    print(f"WARN {f}: year/month/day カラムが見つかりません (旧フォーマット?)")
            except Exception as e:
                print(f"WARN {f}: 日付検証エラー: {e}")
        else:
             print(f"FAIL {f}: カラム順序の不一致。\n検出: {cols[:len(desired_order)]}\n期待: {desired_order}")

if __name__ == "__main__":
    verify_columns()

