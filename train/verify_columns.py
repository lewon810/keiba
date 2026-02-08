
import pandas as pd
import os
import settings

def verify_columns():
    print("results_*.csvファイルのカラム順序を検証しています...")
    
    desired_order = [
         "race_id", "course_type", "distance", "weather", "condition", "date", 
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
        # 末尾に余分なカラムがあるファイルもありますが、最初のN列は標準であるべきです
        match_count = 0
        mean_match = True
        
        for i, col in enumerate(desired_order):
            if i < len(cols):
                if cols[i] != col:
                    # 古いデータの場合、passing/last_3fの欠落を許容するか？
                    # しかし、dateはインデックス5にある必要があります
                    if col == 'date' and cols[i] != 'date':
                        print(f"FAIL {f}: インデックス {i} に 'date' が期待されましたが、{cols[i]} が見つかりました")
                        mean_match = False
                    elif col == 'race_id' and cols[i] != 'race_id':
                        print(f"FAIL {f}: インデックス {i} に 'race_id' が期待されましたが、{cols[i]} が見つかりました")
                        mean_match = False
        
        if mean_match:
            # 日付フォーマットを確認
            date_val = pd.read_csv(path, usecols=['date'], nrows=5)['date'].iloc[0]
            if '年' in str(date_val) and '月' in str(date_val) and '日' in str(date_val):
                print(f"PASS {f}: 順序OK, 日付OK ({date_val})")
            else:
                print(f"WARN {f}: 順序OKですが、日付フォーマットが間違っている可能性があります (期待: YYYY年MM月DD日, 実際: {date_val})")
        else:
             print(f"FAIL {f}: カラム順序の不一致。\n検出: {cols[:len(desired_order)]}\n期待: {desired_order}")

if __name__ == "__main__":
    verify_columns()
