"""
テスト: train.features モジュール（共通定数・ヘルパー関数）

重複コード共通化リファクタリングで新設された features.py の
全関数・定数をテストする。
"""
import pytest
import numpy as np
import pandas as pd
import os
import sys

# プロジェクトルートをパスに追加
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from train.features import (
    FEATURES, PLACE_MAP, PLACE_MAP_SHORT,
    parse_time, get_dist_cat, classify_running_style,
    get_first_position, extract_date_from_race_id,
    lookup_rate, apply_label_encoder, compute_last_3f_features
)


class TestParseTime:
    """parse_time() のテスト"""
    
    def test_minutes_seconds(self):
        assert parse_time('1:34.5') == pytest.approx(94.5)
    
    def test_seconds_only(self):
        assert parse_time('34.5') == pytest.approx(34.5)
    
    def test_zero_minutes(self):
        assert parse_time('0:55.0') == pytest.approx(55.0)
    
    def test_two_minutes(self):
        assert parse_time('2:00.0') == pytest.approx(120.0)
    
    def test_invalid_returns_nan(self):
        assert np.isnan(parse_time('invalid'))
    
    def test_none_returns_nan(self):
        assert np.isnan(parse_time(None))
    
    def test_empty_string_returns_nan(self):
        assert np.isnan(parse_time(''))
    
    def test_numeric_input(self):
        assert parse_time(94.5) == pytest.approx(94.5)


class TestGetDistCat:
    """get_dist_cat() のテスト"""
    
    def test_sprint(self):
        assert get_dist_cat(1200) == 'sprint'
        assert get_dist_cat(1000) == 'sprint'
    
    def test_sprint_boundary(self):
        assert get_dist_cat(1399) == 'sprint'
    
    def test_mile(self):
        assert get_dist_cat(1400) == 'mile'
        assert get_dist_cat(1600) == 'mile'
    
    def test_mile_boundary(self):
        assert get_dist_cat(1899) == 'mile'
    
    def test_intermediate(self):
        assert get_dist_cat(2000) == 'intermediate'
        assert get_dist_cat(2400) == 'intermediate'
    
    def test_intermediate_boundary(self):
        assert get_dist_cat(1900) == 'intermediate'
        assert get_dist_cat(2499) == 'intermediate'
    
    def test_long(self):
        assert get_dist_cat(2500) == 'long'
        assert get_dist_cat(3600) == 'long'
    
    def test_invalid_returns_unknown(self):
        assert get_dist_cat('invalid') == 'unknown'
    
    def test_string_numeric(self):
        assert get_dist_cat('1600') == 'mile'


class TestClassifyRunningStyle:
    """classify_running_style() のテスト"""
    
    def test_front(self):
        assert classify_running_style(1) == "front"
        assert classify_running_style(2) == "front"
    
    def test_middle(self):
        assert classify_running_style(3) == "middle"
        assert classify_running_style(7) == "middle"
    
    def test_back(self):
        assert classify_running_style(8) == "back"
        assert classify_running_style(15) == "back"
        assert classify_running_style(98) == "back"
    
    def test_unknown(self):
        assert classify_running_style(99) == "unknown"
        assert classify_running_style(100) == "unknown"


class TestGetFirstPosition:
    """get_first_position() のテスト"""
    
    def test_normal(self):
        assert get_first_position('3-3-2-1') == 3
    
    def test_first(self):
        assert get_first_position('1-1-1-1') == 1
    
    def test_large_field(self):
        assert get_first_position('12-10-8-5') == 12
    
    def test_no_dash(self):
        assert get_first_position('3') == 99
    
    def test_empty(self):
        assert get_first_position('') == 99
    
    def test_none(self):
        assert get_first_position(None) == 99
    
    def test_non_string(self):
        assert get_first_position(123) == 99


class TestExtractDateFromRaceId:
    """extract_date_from_race_id() のテスト"""
    
    def test_valid_12_digit(self):
        # race_id: YYYY[0:4] PP[4:6] MM[6:8] DD[8:10] XX[10:12]
        # '202505060112' → year=2025, month=06, day=01
        result = extract_date_from_race_id('202505060112')
        assert result == pd.Timestamp('2025-06-01')
    
    def test_valid_long(self):
        result = extract_date_from_race_id('2024120801')
        # 10桁 — 十分な桁がないため NaT
        assert pd.isna(result)
    
    def test_valid_12_digit_december(self):
        # '202505122501' → year=2025, month=12, day=25
        result = extract_date_from_race_id('202505122501')
        assert result == pd.Timestamp('2025-12-25')
    
    def test_short_id_returns_nat(self):
        result = extract_date_from_race_id('12345')
        assert pd.isna(result)
    
    def test_none_returns_nat(self):
        result = extract_date_from_race_id(None)
        assert pd.isna(result)


class TestLookupRate:
    """lookup_rate() のテスト"""
    
    def test_exact_match(self):
        m = {'abc': 0.5}
        assert lookup_rate('abc', m) == 0.5
    
    def test_string_fallback(self):
        m = {'123': 0.3}
        assert lookup_rate(123, m) == 0.3
    
    def test_int_fallback(self):
        m = {456: 0.7}
        assert lookup_rate('456', m) == 0.7
    
    def test_not_found(self):
        m = {'abc': 0.5}
        assert lookup_rate('xyz', m) == 0.0
    
    def test_empty_map(self):
        assert lookup_rate('abc', {}) == 0.0


class TestApplyLabelEncoder:
    """apply_label_encoder() のテスト"""
    
    def test_known_labels(self):
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        le.fit(['a', 'b', 'c'])
        df = pd.DataFrame({'col': ['a', 'b', 'c']})
        df = apply_label_encoder(df, 'col', le)
        assert df['col'].tolist() == le.transform(['a', 'b', 'c']).tolist()
    
    def test_unknown_label_fallback(self):
        """未知のラベルがエラーにならないこと"""
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        le.fit(['a', 'b', 'unknown'])
        df = pd.DataFrame({'col': ['a', 'never_seen']})
        df = apply_label_encoder(df, 'col', le)
        # 'never_seen' は 'unknown' にフォールバックされる
        assert len(df) == 2


class TestComputeLast3fFeatures:
    """compute_last_3f_features() のテスト"""
    
    def test_generates_columns(self):
        df = pd.DataFrame({
            'race_id': ['R1', 'R1', 'R1'],
            'last_3f': [33.5, 34.0, 34.5]
        })
        result = compute_last_3f_features(df)
        assert 'last_3f_time' in result.columns
        assert 'last_3f_rank' in result.columns
        assert 'last_3f_deviation' in result.columns
    
    def test_no_last_3f_column(self):
        df = pd.DataFrame({
            'race_id': ['R1', 'R1'],
            'horse_id': ['H1', 'H2']
        })
        result = compute_last_3f_features(df)
        assert (result['last_3f_time'] == 0).all()
        assert (result['last_3f_rank'] == 99).all()
        assert (result['last_3f_deviation'] == 50).all()


class TestConstants:
    """定数のテスト"""
    
    def test_features_count(self):
        assert len(FEATURES) == 28
    
    def test_features_no_duplicates(self):
        assert len(FEATURES) == len(set(FEATURES))
    
    def test_place_map_has_10_entries(self):
        assert len(PLACE_MAP) == 10
    
    def test_place_map_short_has_10_entries(self):
        assert len(PLACE_MAP_SHORT) == 10
    
    def test_place_map_keys_match(self):
        assert set(PLACE_MAP.keys()) == set(PLACE_MAP_SHORT.keys())
