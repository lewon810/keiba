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
    'jockey_win_rate', 'trainer_win_rate', 'horse_id', 'jockey_id', 'trainer_id',
    'waku', 'umaban', 'course_type', 'distance', 'weather', 'condition',
    'lag1_rank_norm', 'lag1_speed_index', 'lag1_last_3f', 'interval', 'weight_diff',
    'sire_id', 'damsire_id', 'running_style',
    'sire_win_rate', 'damsire_win_rate',
    'course_type_win_rate', 'dist_cat_win_rate',
    'horse_age', 'num_runners',
    'lag2_rank_norm', 'lag3_rank_norm', 'avg_last3_rank_norm',
    # 新特徴量
    'win_streak',           # 直近連勝数（勢い指標）
    'days_since_last_win',  # 前回勝利からの日数（好調期の捕捉）
    # normalized_odds_rank は除外: 逆学習の原因になっていた（不人気馬を勝利馬と誤認）
    'place_win_rate',       # 競馬場別馬勝率（コース適性）
]
# 除外した特徴量と理由:
# - popularity_ratio / popularity: 市場人気のシグナル → オッズに既反映済み
# jockey_win_rate / trainer_win_rate は過去実績ベースの固定値なので残す

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
