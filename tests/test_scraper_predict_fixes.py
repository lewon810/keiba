import pytest
import pandas as pd
import numpy as np
import os
from unittest.mock import MagicMock
from bs4 import BeautifulSoup
from app.scraper import fetch_race_data
from app.history_loader import HistoryLoader
from app import predictor

class MockModel:
    def predict(self, X):
        return np.zeros(len(X))

def test_scraper_course_type_and_popularity(monkeypatch):
    # html for race data
    html = """
    <html>
      <div class="RaceData01">
        14:20発走 / 芝1800m (右 C) / 天候:晴 / 芝:良
      </div>
      <table>
        <tr class="HorseList">
          <td class="Umaban1">1</td>
          <td class="Waku1">1</td>
          <td class="HorseName"><a href="/horse/2023101111/">キタサンブラック</a></td>
          <td class="Jockey"><a href="/jockey/05203/">武豊</a></td>
          <td class="Trainer"><a href="/trainer/01111/">調教師A</a></td>
          <td class="Popular_Ninki">1</td>
          <td class="Popular" id="odds-1">1.5</td>
        </tr>
      </table>
    </html>
    """
    
    class MockResponse:
        def __init__(self, text):
            self.text = text
            self.status_code = 200
            self.encoding = 'utf-8'
            
    def mock_get(url, headers=None, timeout=None):
        return MockResponse(html)
        
    monkeypatch.setattr("requests.get", mock_get)
    
    # odds API をモックして空データを返すようにする
    monkeypatch.setattr("app.scraper.fetch_odds", lambda rid: {})
    
    race_data = fetch_race_data("https://race.netkeiba.com/race/shutuba.html?race_id=202606010809")
    
    assert len(race_data) == 1
    # Bug-1 fix check
    assert race_data[0]["course_type"] == "turf"
    # Issue-2 fix check
    assert race_data[0]["ninki"] == "1"
    assert race_data[0]["popularity"] == "1"

def test_history_loader_date_parsing():
    loader = HistoryLoader()
    # YYYY-MM-DD
    assert loader._parse_date("2026-06-14") == pd.Timestamp("2026-06-14")
    # YYYYMMDD
    assert loader._parse_date("20260614") == pd.Timestamp("2026-06-14")
    # Japanese style (Issue-4 fix check)
    assert loader._parse_date("2026年06月14日") == pd.Timestamp("2026-06-14")
    # Slash style
    assert loader._parse_date("2026/06/14") == pd.Timestamp("2026-06-14")
    # Invalid dates
    assert pd.isna(loader._parse_date("invalid-date"))
    assert pd.isna(loader._parse_date(None))

def test_predictor_fallback_settings_and_profile_merge(monkeypatch, tmp_path):
    # predictor が settings を参照する部分を検証
    # 一時的に horse_profiles.csv を作成する
    raw_dir = tmp_path / "train" / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    
    profile_file = raw_dir / "horse_profiles.csv"
    # Create mock horse_profiles.csv
    profiles_df = pd.DataFrame({
        'horse_id': ['2023101111', '2023102222'],
        'sire_id': ['sire_A', 'sire_B'],
        'damsire_id': ['damsire_A', 'damsire_B']
    })
    profiles_df.to_csv(profile_file, index=False)
    
    # predictor で使用される settings をモック
    class MockSettings:
        MODEL_DIR = str(tmp_path / "train" / "models")
        MODEL_PATH = str(tmp_path / "train" / "models" / "lgbm_ranker_v2.pkl")
        RAW_DATA_DIR = str(raw_dir)
        
    monkeypatch.setattr(predictor, "settings", MockSettings)
    
    # encoders.pkl をモックして作成
    models_dir = tmp_path / "train" / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    
    import joblib
    # encoders.pkl は dictionary
    from sklearn.preprocessing import LabelEncoder
    le_course = LabelEncoder()
    le_course.fit(["turf", "dirt", "steeple", "unknown", "sire_A", "damsire_A"])
    encoders = {
        'jockey_win_rate': {},
        'trainer_win_rate': {},
        'sire_win_rate': {},
        'damsire_win_rate': {},
        'course_type_win_rate': {},
        'dist_cat_win_rate': {},
        'place_win_rate': {},
        'jockey_trainer_win_rate': {},
        'course_runs': {},
        'dist_runs': {},
        'course_type': le_course,
        'weather': le_course,
        'condition': le_course,
        'running_style': le_course,
        'sire_id': le_course,
        'damsire_id': le_course,
        'jockey_id': le_course,
        'trainer_id': le_course,
        'horse_id': le_course,
    }
    joblib.dump(encoders, models_dir / "encoders.pkl")
    
    # モデルのモック
    import joblib
    joblib.dump(MockModel(), models_dir / "lgbm_ranker_v2.pkl")
    
    # Predict を実行
    race_data = [{
        'horse_id': '2023101111',
        'jockey_id': '05203',
        'trainer_id': '01111',
        'course_type': 'turf',
        'distance': '1800',
        'weather': 'sunny',
        'condition': 'good',
        'popularity': '1',
        'odds': '1.5',
        'name': 'キタサンブラック',
        'waku': '1',
        'umaban': '1'
    }]
    
    # history_loader の load をモック
    monkeypatch.setattr("app.history_loader.loader.load", lambda: None)
    monkeypatch.setattr("app.history_loader.loader.get_last_race", lambda h, date: None)
    
    df_result = predictor.predict(race_data, return_df=True)
    
    assert isinstance(df_result, pd.DataFrame)
    # Issue-3 fix check
    assert 'sire_id' in df_result.columns
    assert 'damsire_id' in df_result.columns
