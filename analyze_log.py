import json
import pandas as pd

try:
    with open('evaluate_log.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    df = pd.DataFrame(data)
    
    print('--- Kokura (Place 10) Hits ---')
    kokura_hits = df[(df['place'].str.contains('Kokura|小倉')) & (df['is_hit']==1)].sort_values('return', ascending=False)
    if not kokura_hits.empty:
        print(kokura_hits[['race_id', 'horse_id', 'odds', 'return', 'score']])
    else:
        print("No hits in Kokura.")

    print('\n--- Tokyo (Place 05) Hits ---')
    tokyo_hits = df[(df['place'].str.contains('Tokyo|東京')) & (df['is_hit']==1)].sort_values('return', ascending=False).head(5)
    if not tokyo_hits.empty:
        print(tokyo_hits[['race_id', 'horse_id', 'odds', 'return', 'score']])
    else:
        print("No hits in Tokyo.")

except Exception as e:
    print(f"Error: {e}")
