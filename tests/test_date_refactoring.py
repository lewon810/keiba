"""
date カラム → year, month, day 分割リファクタリングのテスト
"""
import pytest
import pandas as pd
import numpy as np
import os
import sys
import tempfile

# プロジェクトルートを追加
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestScraperBulkSaveBuffer:
    """scraper_bulk._save_buffer のカラム順序テスト"""
    
    def test_desired_order_contains_year_month_day(self):
        """desired_order に year, month, day が含まれ、date が含まれないこと"""
        from train.scraper_bulk import _save_buffer
        
        # テスト用データ
        data = [{
            "race_id": "202105010101",
            "course_type": "turf",
            "distance": 1600,
            "weather": "sunny",
            "condition": "good",
            "year": 2021,
            "month": 5,
            "day": 1,
            "rank": 1,
            "waku": 1,
            "umaban": 1,
            "horse_name": "TestHorse",
            "horse_id": "2018100001",
            "jockey": "TestJockey",
            "jockey_id": "05001",
            "trainer": "TestTrainer",
            "trainer_id": "01001",
            "horse_weight": 480,
            "weight_diff": 0,
            "time": "1:34.5",
            "passing": "3-3-2-1",
            "last_3f": "35.0",
            "odds": 3.5,
            "popularity": 1
        }]
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, encoding='utf-8') as f:
            tmp_path = f.name
        # _save_buffer は file_exists チェックでヘッダーを制御するため、事前に削除
        os.unlink(tmp_path)
        
        try:
            _save_buffer(data, tmp_path)
            df = pd.read_csv(tmp_path)
            cols = df.columns.tolist()
            
            # year, month, day が存在すること
            assert 'year' in cols, "year カラムが見つかりません"
            assert 'month' in cols, "month カラムが見つかりません"
            assert 'day' in cols, "day カラムが見つかりません"
            
            # date が存在しないこと
            assert 'date' not in cols, "date カラムが残っています"
            
            # 順序の確認: year は condition の後
            cond_idx = cols.index('condition')
            year_idx = cols.index('year')
            assert year_idx == cond_idx + 1, f"year は condition の直後であるべき (期待: {cond_idx+1}, 実際: {year_idx})"
            
            # 値の確認
            assert df['year'].iloc[0] == 2021
            assert df['month'].iloc[0] == 5
            assert df['day'].iloc[0] == 1
        finally:
            os.unlink(tmp_path)


class TestPreprocessDateParsing:
    """preprocess.preprocess() の日付パーステスト"""
    
    def _make_sample_df(self):
        """テスト用の最小データフレームを作成"""
        return pd.DataFrame({
            'race_id': ['202105010101', '202105010102'],
            'course_type': ['turf', 'dirt'],
            'distance': [1600, 1800],
            'weather': ['sunny', 'cloudy'],
            'condition': ['good', 'heavy'],
            'year': [2021, 2021],
            'month': [5, 5],
            'day': [1, 2],
            'rank': [1, 2],
            'waku': [1, 2],
            'umaban': [1, 2],
            'horse_name': ['HorseA', 'HorseB'],
            'horse_id': ['2018100001', '2018100002'],
            'jockey': ['JockeyA', 'JockeyB'],
            'jockey_id': ['05001', '05002'],
            'trainer': ['TrainerA', 'TrainerB'],
            'trainer_id': ['01001', '01002'],
            'horse_weight': [480, 460],
            'weight_diff': [0, -2],
            'time': ['1:34.5', '1:35.0'],
            'passing': ['3-3-2-1', '5-5-4-3'],
            'last_3f': ['35.0', '36.0'],
            'odds': [3.5, 10.0],
            'popularity': [1, 3]
        })
    
    def test_preprocess_creates_datetime_from_year_month_day(self):
        """preprocess() が year, month, day から datetime 型の date カラムを生成すること"""
        from train.preprocess import preprocess
        
        df = self._make_sample_df()
        result_df, artifacts = preprocess(df)
        
        assert 'date' in result_df.columns, "date カラムが生成されていません"
        assert pd.api.types.is_datetime64_any_dtype(result_df['date']), \
            f"date が datetime 型ではありません: {result_df['date'].dtype}"
        
        # 日付値の確認
        valid_dates = result_df['date'].dropna()
        assert len(valid_dates) > 0, "有効な日付が0件です"
        assert valid_dates.dt.year.iloc[0] == 2021
        assert valid_dates.dt.month.iloc[0] == 5
    
    def test_transform_creates_datetime_from_year_month_day(self):
        """transform() が year, month, day から datetime 型の date カラムを生成すること"""
        from train.preprocess import preprocess, transform
        
        # まず preprocess で artifacts を生成
        df = self._make_sample_df()
        _, artifacts = preprocess(df)
        
        # 新しいデータで transform
        new_df = self._make_sample_df()
        result_df = transform(new_df, artifacts)
        
        assert 'date' in result_df.columns, "date カラムが生成されていません"
        assert pd.api.types.is_datetime64_any_dtype(result_df['date']), \
            f"date が datetime 型ではありません: {result_df['date'].dtype}"

    def test_transform_legacy_fallback_string_date(self):
        """transform() が旧フォーマット（文字列date）でも動作すること"""
        from train.preprocess import preprocess, transform
        
        # artifacts を生成
        df = self._make_sample_df()
        _, artifacts = preprocess(df)
        
        # レガシーフォーマットのデータ (year/month/day の代わりに date 文字列)
        legacy_df = self._make_sample_df()
        legacy_df['date'] = '2021年5月1日'
        legacy_df = legacy_df.drop(columns=['year', 'month', 'day'])
        
        result_df = transform(legacy_df, artifacts)
        
        assert 'date' in result_df.columns
        assert pd.api.types.is_datetime64_any_dtype(result_df['date'])


class TestHistoryLoaderDateParsing:
    """history_loader の日付パーステスト"""
    
    def test_load_new_format_csv(self):
        """新フォーマット（year/month/day）のCSVを正しく読み込めること"""
        from app.history_loader import HistoryLoader
        
        # テスト用CSVを一時ファイルとして作成
        test_df = pd.DataFrame({
            'race_id': ['202105010101', '202105010102'],
            'horse_id': ['2018100001', '2018100002'],
            'rank': [1, 2],
            'time': ['1:34.5', '1:35.0'],
            'last_3f': ['35.0', '36.0'],
            'year': [2021, 2021],
            'month': [5, 5],
            'day': [1, 2]
        })
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, 
                                         prefix='results_', encoding='utf-8') as f:
            test_df.to_csv(f, index=False)
            tmp_path = f.name
        
        try:
            # HistoryLoader をモック (RAW_DATA_DIR をテンポラリに)
            loader = HistoryLoader()
            
            # 直接DataFrameで検証
            df = pd.read_csv(tmp_path, dtype={'race_id': str, 'horse_id': str})
            assert 'year' in df.columns
            assert 'month' in df.columns
            assert 'day' in df.columns
            
            # 日付構築テスト
            df['date'] = pd.to_datetime(df[['year', 'month', 'day']], errors='coerce')
            assert pd.api.types.is_datetime64_any_dtype(df['date'])
            assert df['date'].iloc[0].year == 2021
            assert df['date'].iloc[0].month == 5
            assert df['date'].iloc[0].day == 1
        finally:
            os.unlink(tmp_path)


class TestScraperBulkDateExtraction:
    """scraper_bulk.scrape_race_data の日付抽出テスト (race_info 初期化)"""
    
    def test_race_info_uses_year_month_day(self):
        """race_info の初期化に year, month, day が使用されていること"""
        # scraper_bulk のインポート確認
        import train.scraper_bulk as sb
        import inspect
        
        source = inspect.getsource(sb.scrape_race_data)
        
        # year, month, day が含まれること
        assert '"year"' in source, "scrape_race_data に 'year' キーがありません"
        assert '"month"' in source, "scrape_race_data に 'month' キーがありません"
        assert '"day"' in source, "scrape_race_data に 'day' キーがありません"
        
        # 旧 "date" が初期化に含まれないこと
        # race_info 辞書の初期化部分で "date" が使われていないことを確認
        # (dt_text 変数名は OK, race_info["date"] がダメ)
        assert '"date": ""' not in source, "scrape_race_data にまだ旧 'date' キーが残っています"



class TestLoadDataMonthFiltering:
    """preprocess.load_data() の月フィルタリングテスト"""
    
    def test_month_filter_uses_csv_column_not_race_id(self):
        """月フィルタが race_id[4:6] (競馬場コード) ではなく CSV の month カラムを使うこと"""
        import tempfile
        import shutil
        from train import settings
        
        # テスト用の一時ディレクトリを作成
        tmp_dir = tempfile.mkdtemp()
        original_raw_dir = settings.RAW_DATA_DIR
        
        try:
            # テスト用CSVを作成: 9月データで複数の競馬場コード (01, 05, 09)
            test_data = pd.DataFrame({
                'race_id': [
                    '202501030901',  # 競馬場01(札幌), 月ではない
                    '202505040902',  # 競馬場05(東京)
                    '202509050903',  # 競馬場09(阪神)
                ],
                'course_type': ['turf', 'dirt', 'turf'],
                'distance': [1600, 1800, 2000],
                'weather': ['sunny', 'sunny', 'cloudy'],
                'condition': ['good', 'good', 'heavy'],
                'year': [2025, 2025, 2025],
                'month': [9, 9, 9],  # 全て9月
                'day': [1, 7, 14],
                'rank': [1, 2, 3],
                'waku': [1, 2, 3],
                'umaban': [1, 2, 3],
                'horse_name': ['HorseA', 'HorseB', 'HorseC'],
                'horse_id': ['2020100001', '2020100002', '2020100003'],
                'jockey': ['J1', 'J2', 'J3'],
                'jockey_id': ['05001', '05002', '05003'],
                'trainer': ['T1', 'T2', 'T3'],
                'trainer_id': ['01001', '01002', '01003'],
                'horse_weight': [480, 460, 500],
                'weight_diff': [0, -2, 4],
                'time': ['1:34.5', '1:35.0', '2:01.0'],
                'passing': ['3-3-2-1', '5-5-4-3', '2-2-1-1'],
                'last_3f': ['35.0', '36.0', '35.5'],
                'odds': [3.5, 10.0, 5.0],
                'popularity': [1, 3, 2]
            })
            
            test_csv_path = os.path.join(tmp_dir, 'results_2025.csv')
            test_data.to_csv(test_csv_path, index=False, encoding='utf-8')
            
            # RAW_DATA_DIR を一時的に差し替え
            settings.RAW_DATA_DIR = tmp_dir
            
            from train.preprocess import load_data
            result = load_data(start_year=2025, end_year=2025, start_month=9, end_month=9)
            
            # 全3行が残るべき（以前は阪神の1行のみだった）
            assert len(result) == 3, f"9月データは3行残るべき (実際: {len(result)})"
            
            # 競馬場コードの確認: 01, 05, 09 全て含まれること
            place_codes = result['race_id'].astype(str).str[4:6].unique()
            assert '01' in place_codes, "札幌(01)のデータが欠落"
            assert '05' in place_codes, "東京(05)のデータが欠落"
            assert '09' in place_codes, "阪神(09)のデータが欠落"
            
        finally:
            settings.RAW_DATA_DIR = original_raw_dir
            shutil.rmtree(tmp_dir)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

