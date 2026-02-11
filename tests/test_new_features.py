import unittest
import pandas as pd
import numpy as np
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from train import preprocess

class TestNewFeatures(unittest.TestCase):
    """上がり3Fとペース特徴量のテスト"""

    def setUp(self):
        """テスト用のダミーデータを作成"""
        self.dummy_data = pd.DataFrame([
            {
                'race_id': '202401010101',
                'horse_id': 'H001',
                'jockey_id': 'J001',
                'trainer_id': 'T001',
                'rank': 1,
                'time': '1:34.5',
                'last_3f': '35.2',
                'passing': '1-1-1-1',
                'course_type': 'turf',
                'distance': 1600,
                'weather': 'sunny',
                'condition': 'good',
                'date': '2024年1月1日',
                'waku': 1,
                'umaban': 1,
                'weight_diff': 0
            },
            {
                'race_id': '202401010101',
                'horse_id': 'H002',
                'jockey_id': 'J002',
                'trainer_id': 'T002',
                'rank': 2,
                'time': '1:34.8',
                'last_3f': '35.5',
                'passing': '2-2-2-2',
                'course_type': 'turf',
                'distance': 1600,
                'weather': 'sunny',
                'condition': 'good',
                'date': '2024年1月1日',
                'waku': 2,
                'umaban': 2,
                'weight_diff': 2
            },
            {
                'race_id': '202401010101',
                'horse_id': 'H003',
                'jockey_id': 'J003',
                'trainer_id': 'T003',
                'rank': 3,
                'time': '1:35.2',
                'last_3f': '36.0',
                'passing': '8-7-6-5',
                'course_type': 'turf',
                'distance': 1600,
                'weather': 'sunny',
                'condition': 'good',
                'date': '2024年1月1日',
                'waku': 3,
                'umaban': 3,
                'weight_diff': -1
            }
        ])

    def test_last_3f_parsing(self):
        """上がり3Fのパース処理を検証"""
        df, _ = preprocess.preprocess(self.dummy_data.copy())
        
        # last_3f_timeが数値として正しくパースされている
        self.assertIn('last_3f_time', df.columns)
        self.assertEqual(df.iloc[0]['last_3f_time'], 35.2)
        self.assertEqual(df.iloc[1]['last_3f_time'], 35.5)
        self.assertEqual(df.iloc[2]['last_3f_time'], 36.0)

    def test_last_3f_rank_calculation(self):
        """レース内順位計算を検証"""
        df, _ = preprocess.preprocess(self.dummy_data.copy())
        
        # last_3f_rankが正しく計算されている（最速馬が1位）
        self.assertIn('last_3f_rank', df.columns)
        self.assertEqual(df.iloc[0]['last_3f_rank'], 1)  # 35.2秒が最速
        self.assertEqual(df.iloc[1]['last_3f_rank'], 2)  # 35.5秒が2番目
        self.assertEqual(df.iloc[2]['last_3f_rank'], 3)  # 36.0秒が3番目

    def test_last_3f_deviation(self):
        """偏差値計算を検証"""
        df, _ = preprocess.preprocess(self.dummy_data.copy())
        
        # last_3f_deviationが存在し、平均値付近に分布
        self.assertIn('last_3f_deviation', df.columns)
        mean_deviation = df['last_3f_deviation'].mean()
        
        # 偏差値の平均は約50になるはず
        self.assertAlmostEqual(mean_deviation, 50, delta=15)
        
        # 最速馬（last_3f_time最小）は偏差値が高い（>50）
        fastest_idx = df['last_3f_time'].idxmin()
        self.assertGreater(df.loc[fastest_idx, 'last_3f_deviation'], 50)

    def test_pace_calculation(self):
        """ペース情報集計を検証"""
        df, _ = preprocess.preprocess(self.dummy_data.copy())
        
        # front_runner_countが存在
        self.assertIn('front_runner_count', df.columns)
        self.assertIn('pace_ratio', df.columns)
        
        # このレースでは2頭が先行（passing 1コーナー≤2位）
        self.assertEqual(df.iloc[0]['front_runner_count'], 2)
        
        # レース内3頭中2頭が先行、ratio = 2/3
        expected_ratio = 2 / 3
        self.assertAlmostEqual(df.iloc[0]['pace_ratio'], expected_ratio, places=2)

    def test_missing_last_3f_column(self):
        """last_3fカラムが無い場合のデフォルト値を検証"""
        df_no_3f = self.dummy_data.drop(columns=['last_3f']).copy()
        df, _ = preprocess.preprocess(df_no_3f)
        
        # デフォルト値が設定されている
        self.assertEqual(df.iloc[0]['last_3f_time'], 0)
        self.assertEqual(df.iloc[0]['last_3f_rank'], 99)
        self.assertEqual(df.iloc[0]['last_3f_deviation'], 50)

    def test_missing_passing_column(self):
        """passingカラムが無い場合のデフォルト値を検証"""
        df_no_passing = self.dummy_data.drop(columns=['passing']).copy()
        df, _ = preprocess.preprocess(df_no_passing)
        
        # デフォルト値が設定されている
        self.assertEqual(df.iloc[0]['front_runner_count'], 0)
        self.assertEqual(df.iloc[0]['pace_ratio'], 0)

    def test_transform_function(self):
        """transform関数で同じ特徴量が生成されることを検証"""
        # まずpreprocessで学習用データとartifactsを作成
        df_train, artifacts = preprocess.preprocess(self.dummy_data.copy())
        
        # transformで推論用データを処理
        df_test = preprocess.transform(self.dummy_data.copy(), artifacts)
        
        # 同じ特徴量が存在する
        for col in ['last_3f_time', 'last_3f_rank', 'last_3f_deviation', 
                    'front_runner_count', 'pace_ratio']:
            self.assertIn(col, df_test.columns, f"{col} should exist in transform output")

if __name__ == '__main__':
    unittest.main()
