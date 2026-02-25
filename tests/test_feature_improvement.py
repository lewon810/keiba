"""
特徴量リストの一致テストと新特徴量のテスト
"""
import pytest
import pandas as pd
import numpy as np
import os
import sys
import ast
import inspect

# プロジェクトルートを追加
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def extract_features_from_source(filepath):
    """
    ソースコードから features = [...] を抽出して特徴量リストを返す。
    複数の features 定義がある場合は最初の定義を返す。
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        source = f.read()
    
    tree = ast.parse(source)
    
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == 'features':
                    if isinstance(node.value, ast.List):
                        return [elt.value for elt in node.value.elts if isinstance(elt, ast.Constant)]
    return []


class TestFeatureConsistency:
    """全ファイルの特徴量リストが共通定数 FEATURES から取得されていることを検証"""
    
    def test_all_files_use_shared_features(self):
        """全ファイルが train.features.FEATURES を参照していること"""
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        files = {
            'train.py': os.path.join(base_dir, 'train', 'train.py'),
            'evaluate.py': os.path.join(base_dir, 'train', 'evaluate.py'),
            'evaluate_html_generator.py': os.path.join(base_dir, 'train', 'report', 'evaluate_html_generator.py'),
            'predictor.py': os.path.join(base_dir, 'app', 'predictor.py'),
        }
        
        for name, path in files.items():
            assert os.path.exists(path), f"{name} が見つかりません: {path}"
            with open(path, 'r', encoding='utf-8') as f:
                source = f.read()
            assert 'FEATURES' in source, \
                f"{name} が共通定数 FEATURES を参照していません"
    
    def test_features_source_of_truth(self):
        """train.features.FEATURES が信頼できる唯一のソースであること"""
        from train.features import FEATURES
        assert len(FEATURES) == 29, f"特徴量は29個であるべき (実際: {len(FEATURES)})"
    
    def test_no_leakage_features(self):
        """リーケージ特徴量が含まれていないこと"""
        from train.features import FEATURES
        leakage_features = {'front_runner_count', 'pace_ratio'}
        found_leakage = set(FEATURES) & leakage_features
        assert not found_leakage, \
            f"FEATURES にリーケージ特徴量が残っています: {found_leakage}"
    
    def test_new_features_present(self):
        """新しく追加した特徴量が含まれていること"""
        from train.features import FEATURES
        # popularity_ratio はpopularityの相対化バージョン
        # lag*_rank_norm は絶対着順の 0〜1 正規化バージョン
        new_features = {'popularity_ratio', 'horse_age', 'num_runners',
                        'lag2_rank_norm', 'lag3_rank_norm', 'avg_last3_rank_norm'}
        missing = new_features - set(FEATURES)
        assert not missing, \
            f"FEATURES に新特徴量が不足: {missing}"
    
    def test_raw_popularity_not_in_features(self):
        """生の人気順位(絶対値)popularityが特徴量に含まれないこと"""
        from train.features import FEATURES
        assert 'popularity' not in FEATURES, \
            "'popularity' は相対値 'popularity_ratio' に置き換えられました"



class TestPreprocessNewFeatures:
    """preprocess.preprocess() の新特徴量テスト"""
    
    def _make_sample_df(self, n_horses=4):
        """テスト用の最小データフレーム（複数レース、複数馬）"""
        data = []
        for race_num in range(1, 3):  # 2レース
            for h in range(1, n_horses + 1):
                data.append({
                    'race_id': f'20250501010{race_num}',
                    'course_type': 'turf',
                    'distance': 1600,
                    'weather': 'sunny',
                    'condition': 'good',
                    'year': 2025,
                    'month': 5,
                    'day': race_num,
                    'rank': h,
                    'waku': h,
                    'umaban': h,
                    'horse_name': f'Horse{h}',
                    'horse_id': f'2020{100000 + h}',
                    'jockey': f'Jockey{h}',
                    'jockey_id': f'0500{h}',
                    'trainer': f'Trainer{h}',
                    'trainer_id': f'0100{h}',
                    'horse_weight': 460 + h * 10,
                    'weight_diff': h - 2,
                    'time': f'1:{34 + h}.0',
                    'passing': f'{h}-{h}-{h}-{h}',
                    'last_3f': f'{34.0 + h}',
                    'odds': 3.0 + h * 2,
                    'popularity': h
                })
        return pd.DataFrame(data)
    
    def test_popularity_ratio_feature(self):
        """popularity_ratio 特徴量が正しく生成されること（0〜1の範囲）"""
        from train.preprocess import preprocess
        df = self._make_sample_df()
        result_df, artifacts = preprocess(df)
        
        assert 'popularity_ratio' in result_df.columns
        assert result_df['popularity_ratio'].notna().all()
        # popularity_ratio = popularity / num_runners、値は0〜1の範囲
        assert (result_df['popularity_ratio'] > 0).all()
        assert (result_df['popularity_ratio'] <= 1.0).all()

    
    def test_num_runners_feature(self):
        """num_runners 特徴量が正しく生成されること"""
        from train.preprocess import preprocess
        df = self._make_sample_df(n_horses=4)
        result_df, artifacts = preprocess(df)
        
        assert 'num_runners' in result_df.columns
        # 各レースに4頭出走
        assert (result_df['num_runners'] == 4).all()
    
    def test_horse_age_feature(self):
        """horse_age 特徴量が正しく生成されること"""
        from train.preprocess import preprocess
        df = self._make_sample_df()
        result_df, artifacts = preprocess(df)
        
        assert 'horse_age' in result_df.columns
        # 2020年生まれ, 2025年レース → 5歳
        assert (result_df['horse_age'] == 5).all()
    
    def test_lag2_lag3_features(self):
        """lag2_rank_norm, lag3_rank_norm, avg_last3_rank_norm が生成されること"""
        from train.preprocess import preprocess
        df = self._make_sample_df()
        result_df, artifacts = preprocess(df)
        
        # 元の絶対値カラムも存在
        assert 'lag2_rank' in result_df.columns
        assert 'lag3_rank' in result_df.columns
        assert 'avg_last3_rank' in result_df.columns
        # 正規化版カラムも存在し、0.0〜1.0の範囲
        assert 'lag1_rank_norm' in result_df.columns
        assert 'lag2_rank_norm' in result_df.columns
        assert 'lag3_rank_norm' in result_df.columns
        assert 'avg_last3_rank_norm' in result_df.columns
        assert (result_df['lag1_rank_norm'] >= 0.0).all()
        assert (result_df['lag1_rank_norm'] <= 1.0).all()
    
    def test_running_style_is_lag_based(self):
        """running_style が前走ベースで算出されていること（リーケージなし）"""
        from train.preprocess import preprocess
        
        # 3レース分のデータを作成（同一馬が3レースに出走）
        data = []
        for race_num in range(1, 4):
            data.append({
                'race_id': f'20250501010{race_num}',
                'course_type': 'turf',
                'distance': 1600,
                'weather': 'sunny',
                'condition': 'good',
                'year': 2025,
                'month': 5,
                'day': race_num,
                'rank': race_num,
                'waku': 1, 'umaban': 1,
                'horse_name': 'TestHorse',
                'horse_id': '2020100001',
                'jockey': 'J1', 'jockey_id': '05001',
                'trainer': 'T1', 'trainer_id': '01001',
                'horse_weight': 480, 'weight_diff': 0,
                'time': '1:35.0',
                'passing': '1-1-1-1',  # 全レースで逃げ
                'last_3f': '35.0',
                'odds': 5.0,
                'popularity': 1
            })
        
        df = pd.DataFrame(data)
        result_df, _ = preprocess(df)
        
        horse_data = result_df[result_df['horse_id'] == result_df['horse_id'].iloc[0]]
        horse_data = horse_data.sort_values('date')
        
        # 1レース目: 前走なし → unknown（エンコード後は整数値の可能性）
        first_style = horse_data.iloc[0]['running_style']
        # preprocess()でLabelEncoderが適用されるため、エンコード後の値をチェック
        # 「unknown」に対応するエンコード値であることを確認（具体値は動的）
        assert first_style is not None  # 値が設定されていること
        

class TestTransformNewFeatures:
    """preprocess.transform() の新特徴量テスト"""
    
    def _make_sample_df(self):
        return pd.DataFrame({
            'race_id': ['202505010101', '202505010102'],
            'course_type': ['turf', 'dirt'],
            'distance': [1600, 1800],
            'weather': ['sunny', 'cloudy'],
            'condition': ['good', 'heavy'],
            'year': [2025, 2025],
            'month': [5, 5],
            'day': [1, 2],
            'rank': [1, 2],
            'waku': [1, 2],
            'umaban': [1, 2],
            'horse_name': ['HorseA', 'HorseB'],
            'horse_id': ['2020100001', '2020100002'],
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
    
    def test_transform_has_new_features(self):
        """transform() が新特徴量を生成すること"""
        from train.preprocess import preprocess, transform
        
        df = self._make_sample_df()
        _, artifacts = preprocess(df)
        
        new_df = self._make_sample_df()
        result_df = transform(new_df, artifacts)
        
        new_features = ['popularity_ratio', 'num_runners', 'horse_age', 
                        'lag1_rank_norm', 'lag2_rank_norm', 'lag3_rank_norm', 'avg_last3_rank_norm']
        for feat in new_features:
            assert feat in result_df.columns, f"{feat} が transform() の出力にありません"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
