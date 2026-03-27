import numpy as np
import pytest
import pandas as pd
from app.history_loader import loader
from app.predictor import predict


def test_history_loader_features():
    """history_loader が lag1_distance と lag1_popularity を正しく返すことを確認する。"""
    loader.df = pd.DataFrame([
        {'horse_id': '123', 'date': pd.to_datetime('2023-01-01'), 'rank': 1,
         'speed_index': 50, 'last_3f': 35.5, 'distance': 1600, 'popularity': 2},
        {'horse_id': '123', 'date': pd.to_datetime('2023-02-01'), 'rank': 2,
         'speed_index': 52, 'last_3f': 34.0, 'distance': 1800, 'popularity': 1}
    ])
    loader.is_loaded = True

    last_stats = loader.get_last_race('123', '2023-03-01')
    assert last_stats is not None
    assert 'lag1_distance' in last_stats, "lag1_distance が get_last_race() に含まれていません"
    assert 'lag1_popularity' in last_stats, "lag1_popularity が get_last_race() に含まれていません"
    assert last_stats['lag1_distance'] == 1800
    assert last_stats['lag1_popularity'] == 1


def test_predictor_missing_columns(monkeypatch):
    """lag1_distance_diff と lag1_popularity が predictor の出力 DataFrame に含まれることを確認する。"""
    class MockModel:
        def predict(self, X):
            return np.zeros(len(X))

    def mock_load(path):
        if 'encoder' in path:
            return {}  # encoders.pkl → 空の dict
        return MockModel()  # それ以外（model file）は MockModel

    import app.predictor
    monkeypatch.setattr(app.predictor.os.path, 'exists', lambda p: True)
    monkeypatch.setattr(app.predictor.joblib, 'load', mock_load)

    # history_loader に前走データをセット
    loader.df = pd.DataFrame([{
        'horse_id': '123',
        'date': pd.to_datetime('2023-01-01'),
        'rank': 1,
        'speed_index': 50,
        'last_3f': 35.5,
        'distance': 1600,
        'popularity': 2
    }])
    loader.is_loaded = True

    race_data = [{
        'horse_id': '123',
        'date': '2023-03-01',
        'name': 'テスト馬',
        'jockey_id': '1',
        'trainer_id': '1',
        'course_type': 'turf',
        'weather': 'sunny',
        'condition': 'good',
        'sire_id': '1',
        'damsire_id': '1',
        'running_style': 'front',
        'waku': 1,
        'umaban': 1,
        'distance': 2000,
        'weight_diff': 0,
        'popularity': 3,
        'odds': 5.0
    }]

    res = predict(race_data, return_df=True)
    assert isinstance(res, pd.DataFrame), f"predict() は DataFrame を返すべきですが、got: {type(res)}: {res}"
    assert 'lag1_distance_diff' in res.columns, "lag1_distance_diff が特徴量列に含まれていません"
    assert 'lag1_popularity' in res.columns, "lag1_popularity が特徴量列に含まれていません"
    # lag1_distance=1600, current distance=2000 -> diff=400
    assert res['lag1_distance_diff'].iloc[0] == 400, \
        f"lag1_distance_diff が期待値 400 ではなく {res['lag1_distance_diff'].iloc[0]} でした"
    assert res['lag1_popularity'].iloc[0] == 2, \
        f"lag1_popularity が期待値 2 ではなく {res['lag1_popularity'].iloc[0]} でした"
