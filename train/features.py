"""
共通定数・ヘルパー関数モジュール。

複数のスクリプト（preprocess.py, predictor.py, history_loader.py, evaluate.py 等）で
重複していたロジックをここに集約し、修正漏れを防止する。
"""
import numpy as np
import pandas as pd


# ===================================================
# 定数
# ===================================================

# モデルが使用する特徴量リスト（学習・推論・評価で共通）
FEATURES = [
    'jockey_win_rate', 'trainer_win_rate',
    'waku', 'umaban', 'course_type', 'distance', 'weather', 'condition',
    'lag1_rank_norm', 'lag1_speed_index', 'lag1_last_3f', 'interval', 'weight_diff',
    'running_style',
    'sire_win_rate', 'damsire_win_rate',
    'course_type_win_rate', 'dist_cat_win_rate',
    'horse_age', 'num_runners',
    'lag2_rank_norm', 'lag3_rank_norm', 'avg_last3_rank_norm',
    # 連勝・好調期
    'win_streak',           # 直近連勝数（勢い指標）
    'days_since_last_win',  # 前回勝利からの日数（好調期の捕捉）
    # normalized_odds_rank は除外: 逆学習の原因になっていた（不人気馬を勝利馬と誤認）
    'place_win_rate',       # 競馬場別馬勝率（コース適性）
    # スピード指数トレンド（新特徴量）
    'lag2_speed_index',          # 2走前スピード指数
    'avg_last3_speed_index',     # 直近3走スピード指数平均
    'speed_trend',               # スピード指数トレンド（lag1 - avg_last3）
    # 騎手×調教師コンビ勝率（新特徴量）
    'jockey_trainer_combo_win_rate',
    # 新規追加特徴量
    'is_long_rest',
    'is_first_course_type',
    'is_first_dist_cat',
    'lag1_distance_diff',
    'lag1_popularity',
]
# 除外した特徴量と理由:
# - horse_id / jockey_id / trainer_id / sire_id / damsire_id:
#   高カーディナリティIDはLabelEncoderで連番化しても意味がない。
#   各IDの予測シグナルは勝率マップ（win_rate系特徴量）で十分に捉えている。
# - popularity_ratio / popularity: 市場人気のシグナル → オッズに既反映済み

# 競馬場コード → 名称マッピング
PLACE_MAP = {
    "01": "札幌 (Sapporo)", "02": "函館 (Hakodate)",
    "03": "福島 (Fukushima)", "04": "新潟 (Niigata)",
    "05": "東京 (Tokyo)", "06": "中山 (Nakayama)",
    "07": "中京 (Chukyo)", "08": "京都 (Kyoto)",
    "09": "阪神 (Hanshin)", "10": "小倉 (Kokura)"
}

# evaluate_html_generator 用の短縮版（英語のみ）
PLACE_MAP_SHORT = {
    "01": "Sapporo", "02": "Hakodate", "03": "Fukushima", "04": "Niigata",
    "05": "Tokyo", "06": "Nakayama", "07": "Chukyo", "08": "Kyoto",
    "09": "Hanshin", "10": "Kokura"
}


# ===================================================
# ヘルパー関数
# ===================================================

def parse_time(t_str):
    """タイム文字列を秒に変換する。例: '1:34.5' → 94.5"""
    try:
        t_str = str(t_str)
        if ':' in t_str:
            m, s = t_str.split(':')
            return int(m) * 60 + float(s)
        return float(t_str)
    except:
        return np.nan


def get_dist_cat(d):
    """距離からカテゴリを返す。Sprint / Mile / Intermediate / Long"""
    try:
        d = int(d)
        if d < 1400: return 'sprint'
        if d < 1900: return 'mile'
        if d < 2500: return 'intermediate'
        return 'long'
    except:
        return 'unknown'


def classify_running_style(pos):
    """通過順位から脚質を分類する。"""
    if pos <= 2: return "front"    # 逃げ・先行
    if pos <= 7: return "middle"   # 先行・差し
    if pos < 99: return "back"     # 差し・追込
    return "unknown"


def get_first_position(passing):
    """通過順文字列（例: '3-3-2-1'）から最初の位置を整数で返す。"""
    if not passing or not isinstance(passing, str) or '-' not in passing:
        return 99
    try:
        pos_list = [int(p) for p in passing.split('-') if p.isdigit()]
        return pos_list[0] if pos_list else 99
    except:
        return 99


def extract_date_from_race_id(rid):
    """race_id（12桁以上）から日付を抽出して datetime を返す。"""
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


def lookup_rate(key, rate_map):
    """
    win_rate マップから安全にレート値を取得する。
    キーの型不一致（int/str）にも対応。
    """
    if key in rate_map:
        return rate_map[key]
    str_key = str(key)
    if str_key in rate_map:
        return rate_map[str_key]
    try:
        int_key = int(key)
        if int_key in rate_map:
            return rate_map[int_key]
    except (ValueError, TypeError):
        pass
    return 0.0


def apply_label_encoder(df, col, le):
    """
    DataFrame のカラムに LabelEncoder を安全に適用する。
    未知のラベルは 'unknown' にフォールバックし、それも未知なら
    学習済みクラスの先頭値にフォールバックする。
    """
    valid_classes = set(le.classes_)
    df[col] = df[col].astype(str).map(lambda x: x if x in valid_classes else "unknown")
    if "unknown" not in valid_classes:
        fallback = list(valid_classes)[0]
        df[col] = df[col].map(lambda x: x if x in valid_classes else fallback)
    df[col] = le.transform(df[col]).astype(int)
    return df


def compute_last_3f_features(df):
    """
    上がり3ハロン関連の特徴量を算出する。
    入力 DataFrame に last_3f カラムがある場合:
      - last_3f_time: 数値変換
      - last_3f_rank: レース内順位
      - last_3f_deviation: 偏差値（50基準）
    """
    if 'last_3f' in df.columns:
        df['last_3f_time'] = pd.to_numeric(df['last_3f'], errors='coerce').fillna(0)

        # レース内順位
        df['last_3f_rank'] = df.groupby('race_id')['last_3f_time'].rank(
            method='min', ascending=True
        ).fillna(99)

        # 偏差値
        race_3f_stats = df.groupby('race_id')['last_3f_time'].agg(['mean', 'std']).reset_index()
        race_3f_stats.columns = ['race_id', 'race_3f_mean', 'race_3f_std']
        df = df.merge(race_3f_stats, on='race_id', how='left')

        df['last_3f_deviation'] = 50 - (
            (df['last_3f_time'] - df['race_3f_mean']) / df['race_3f_std'].replace(0, 1)
        ) * 10
        df['last_3f_deviation'] = df['last_3f_deviation'].fillna(50)

        df = df.drop(columns=['race_3f_mean', 'race_3f_std'], errors='ignore')
    else:
        df['last_3f_time'] = 0
        df['last_3f_rank'] = 99
        df['last_3f_deviation'] = 50

    return df


def apply_artifacts_to_df(df, artifacts):
    """
    artifacts（学習済みマップ）を使ってDataFrameに特徴量を付与する共通関数。
    predictor.py と preprocess.transform() の両方から呼び出す。
    
    前提: df に以下が計算済みであること
      - lag1_rank, lag2_rank, lag3_rank, avg_last3_rank  (lag features)
      - horse_id, jockey_id, trainer_id, course_type, distance 等の基本列
    """
    # 1. Rank Normalization (rank_norm系)
    if 'lag1_rank' in df.columns:
        df['lag1_rank_norm'] = (df['lag1_rank'].clip(1, 10) - 1) / 9.0
        df['lag2_rank_norm'] = (df['lag2_rank'].clip(1, 10) - 1) / 9.0
        df['lag3_rank_norm'] = (df['lag3_rank'].clip(1, 10) - 1) / 9.0
        df['avg_last3_rank_norm'] = (df['avg_last3_rank'].clip(1, 10) - 1) / 9.0

    # 2. Win Rate Maps (Jockey, Trainer, Sire, DamSire)
    encoding_cols = [
        ('jockey_win_rate', 'jockey_id'),
        ('trainer_win_rate', 'trainer_id'),
        ('sire_win_rate', 'sire_id'),
        ('damsire_win_rate', 'damsire_id')
    ]
    for col, id_col in encoding_cols:
        if col in artifacts and col not in df.columns:
            map_dict = artifacts[col]
            if id_col in df.columns:
                df[col] = df[id_col].astype(str).apply(lambda k: lookup_rate(k, map_dict))
            else:
                df[col] = 0.0
        elif col not in df.columns:
            df[col] = 0.0

    # 3. Aptitude Maps (Course Type, Distance)
    if 'aptitude_type' in artifacts and 'course_type_win_rate' not in df.columns:
        type_map = artifacts['aptitude_type']
        def _get_type_aptitude(row):
            hid = str(row.get('horse_id', ''))
            ctype = row.get('course_type', 'unknown')
            if hid in type_map and ctype in type_map[hid]:
                return type_map[hid][ctype]
            return 0.0
        df['course_type_win_rate'] = df.apply(_get_type_aptitude, axis=1)
    elif 'course_type_win_rate' not in df.columns:
        df['course_type_win_rate'] = 0.0

    if 'aptitude_dist' in artifacts and 'dist_cat_win_rate' not in df.columns:
        dist_map = artifacts['aptitude_dist']
        if 'distance' in df.columns and 'dist_cat' not in df.columns:
            df['dist_cat'] = df['distance'].apply(get_dist_cat)
        def _get_dist_aptitude(row):
            hid = str(row.get('horse_id', ''))
            cat = row.get('dist_cat', 'unknown')
            if hid in dist_map and cat in dist_map[hid]:
                return dist_map[hid][cat]
            return 0.0
        df['dist_cat_win_rate'] = df.apply(_get_dist_aptitude, axis=1)
    elif 'dist_cat_win_rate' not in df.columns:
        df['dist_cat_win_rate'] = 0.0

    # 4. Place Win Rate
    if 'place_win_rate' in artifacts and 'place_win_rate' not in df.columns:
        pw_map = artifacts['place_win_rate']
        def _get_place_win_rate(row):
            hid = str(row.get('horse_id', ''))
            if 'place_code' in row and pd.notna(row['place_code']):
                pc = str(row['place_code'])
            elif 'race_id' in row and pd.notna(row['race_id']):
                pc = str(row['race_id'])[4:6]
            else:
                pc = 'unknown'
            if hid in pw_map and pc in pw_map[hid]:
                return pw_map[hid][pc]
            return 0.0
        df['place_win_rate'] = df.apply(_get_place_win_rate, axis=1)
    elif 'place_win_rate' not in df.columns:
        df['place_win_rate'] = 0.0

    # 5. Jockey × Trainer Combo Win Rate
    if 'jockey_trainer_win_rate' in artifacts and 'jockey_trainer_combo_win_rate' not in df.columns:
        combo_map = artifacts['jockey_trainer_win_rate']
        def _get_combo_win_rate(row):
            key = str(row.get('jockey_id', 'unknown')) + '_' + str(row.get('trainer_id', 'unknown'))
            return lookup_rate(key, combo_map)
        df['jockey_trainer_combo_win_rate'] = df.apply(_get_combo_win_rate, axis=1)
    elif 'jockey_trainer_combo_win_rate' not in df.columns:
        df['jockey_trainer_combo_win_rate'] = 0.0

    # 6. Default Fallbacks and New Features 
    if 'win_streak' not in df.columns:
        df['win_streak'] = 0
    if 'days_since_last_win' not in df.columns:
        df['days_since_last_win'] = 365
        
    if 'interval' in df.columns:
        df['is_long_rest'] = (df['interval'] >= 90).astype(int)
    else:
        df['is_long_rest'] = 0
        
    # 推論時のみ適用される（学習時には既に充当済み）
    if 'course_runs' in artifacts and 'is_first_course_type' not in df.columns:
        runs_map = artifacts['course_runs']
        def _get_type_runs(row):
            hid = str(row.get('horse_id', ''))
            ctype = row.get('course_type', 'unknown')
            if hid in runs_map and ctype in runs_map[hid]:
                return runs_map[hid][ctype]
            return 0
        df['is_first_course_type'] = (df.apply(_get_type_runs, axis=1) == 0).astype(int)
    elif 'is_first_course_type' not in df.columns:
        df['is_first_course_type'] = 0

    if 'dist_runs' in artifacts and 'is_first_dist_cat' not in df.columns:
        runs_map = artifacts['dist_runs']
        if 'distance' in df.columns and 'dist_cat' not in df.columns:
            df['dist_cat'] = df['distance'].apply(get_dist_cat)
        def _get_dist_runs(row):
            hid = str(row.get('horse_id', ''))
            cat = row.get('dist_cat', 'unknown')
            if hid in runs_map and cat in runs_map[hid]:
                return runs_map[hid][cat]
            return 0
        df['is_first_dist_cat'] = (df.apply(_get_dist_runs, axis=1) == 0).astype(int)
    elif 'is_first_dist_cat' not in df.columns:
        df['is_first_dist_cat'] = 0
        
    # 7. Label Encoding
    # artifacts内のLabelEncoderオブジェクトを判別して適用する
    from sklearn.preprocessing import LabelEncoder
    for col, obj in artifacts.items():
        if isinstance(obj, LabelEncoder) and col in df.columns:
            df = apply_label_encoder(df, col, obj)

    return df


def compute_score(df, win_prob_col='win_prob', odds_col='odds'):
    """
    スコア計算の共通関数（evaluate / predict 共通）。
    score = win_prob × odds（期待値ベース）。
    オッズが 0 の場合は win_prob をそのまま使用する。
    """
    odds = pd.to_numeric(df[odds_col], errors='coerce').fillna(0)
    prob = df[win_prob_col]
    score = np.where(odds > 0, prob * odds, prob)
    return score


