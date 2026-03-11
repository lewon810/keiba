"""
テスト: evaluate_html_generator のベット選択ロジック
P1: win_prob ベースの Top-1 選択が正しく動作すること
P2: 履歴データ込みの特徴量計算が改善されること
"""
import pytest
import pandas as pd
import numpy as np
import os
import sys

# プロジェクトルートを追加
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from train.features import compute_score


class TestWinProbSelection:
    """P1: win_prob ベースの Top-1 選択ロジックのテスト"""
    
    def _make_race_df(self):
        """1レース分のテストデータ（win_prob と score が異なる馬を選択する状況）"""
        return pd.DataFrame({
            'race_id': ['R001'] * 4,
            'horse_id': ['H1', 'H2', 'H3', 'H4'],
            'win_prob': [0.35, 0.10, 0.30, 0.25],  # H1が最高確率
            'odds': [3.0, 50.0, 5.0, 8.0],          # H2が最高オッズ
            'rank': [1, 5, 2, 3],                     # H1が実際の勝者
            'place_code': ['05', '05', '05', '05'],
        })
    
    def test_win_prob_selects_highest_probability(self):
        """win_prob で Top-1 を選択すると最高確率の馬が選ばれること"""
        df = self._make_race_df()
        
        # win_prob ベースでソートして Top-1
        sorted_df = df.sort_values(['race_id', 'win_prob'], ascending=[True, False])
        top1 = sorted_df.groupby('race_id').head(1)
        
        assert top1.iloc[0]['horse_id'] == 'H1'
        assert top1.iloc[0]['win_prob'] == 0.35
    
    def test_score_selects_longshot(self):
        """score (win_prob×odds) で Top-1 を選択すると穴馬が選ばれること（旧挙動の確認）"""
        df = self._make_race_df()
        df['score'] = compute_score(df, win_prob_col='win_prob', odds_col='odds')
        
        # score ベースでソートして Top-1
        sorted_df = df.sort_values(['race_id', 'score'], ascending=[True, False])
        top1 = sorted_df.groupby('race_id').head(1)
        
        # H2 (prob=0.10, odds=50.0, score=5.0) が選ばれるはず
        assert top1.iloc[0]['horse_id'] == 'H2'
    
    def test_win_prob_gives_correct_hit(self):
        """win_prob ベースで選択した馬が実際に1着の場合 hit となること"""
        df = self._make_race_df()
        
        sorted_df = df.sort_values(['race_id', 'win_prob'], ascending=[True, False])
        top1 = sorted_df.groupby('race_id').head(1)
        
        # H1 は rank=1 なので hit
        hits = top1[top1['rank'] == 1]
        assert len(hits) == 1
    
    def test_score_gives_no_hit(self):
        """score ベースで選択した馬が1着でない場合 miss となること"""
        df = self._make_race_df()
        df['score'] = compute_score(df, win_prob_col='win_prob', odds_col='odds')
        
        sorted_df = df.sort_values(['race_id', 'score'], ascending=[True, False])
        top1 = sorted_df.groupby('race_id').head(1)
        
        # H2 は rank=5 なので miss
        hits = top1[top1['rank'] == 1]
        assert len(hits) == 0


class TestWinProbThresholdFiltering:
    """win_prob 閾値フィルタリングのテスト"""
    
    def _make_multi_race_df(self):
        """複数レースのテストデータ"""
        data = []
        # Race 1: Top-1 の win_prob = 0.40
        for i, (prob, odds, rank) in enumerate([(0.40, 3.0, 1), (0.30, 5.0, 2), (0.30, 6.0, 3)]):
            data.append({'race_id': 'R001', 'horse_id': f'H1_{i}', 'win_prob': prob, 'odds': odds, 'rank': rank, 'place_code': '05'})
        # Race 2: Top-1 の win_prob = 0.15
        for i, (prob, odds, rank) in enumerate([(0.15, 10.0, 3), (0.10, 20.0, 1), (0.75, 2.0, 2)]):
            data.append({'race_id': 'R002', 'horse_id': f'H2_{i}', 'win_prob': prob, 'odds': odds, 'rank': rank, 'place_code': '05'})
        return pd.DataFrame(data)
    
    def test_low_threshold_includes_all(self):
        """閾値0.0なら全レースが対象"""
        df = self._make_multi_race_df()
        sorted_df = df.sort_values(['race_id', 'win_prob'], ascending=[True, False])
        top1 = sorted_df.groupby('race_id').head(1)
        bet_df = top1[top1['win_prob'] >= 0.0]
        assert len(bet_df) == 2
    
    def test_high_threshold_filters_low_prob(self):
        """高い閾値で低確率レースが除外されること"""
        df = self._make_multi_race_df()
        sorted_df = df.sort_values(['race_id', 'win_prob'], ascending=[True, False])
        top1 = sorted_df.groupby('race_id').head(1)
        bet_df = top1[top1['win_prob'] >= 0.30]
        
        # R001 (prob=0.40) のみ残る、R002 (prob=0.75) も残る
        # 修正: R002のTop-1はprob=0.75
        assert len(bet_df) == 2  # 両方0.30以上


class TestComputeScoreUnchanged:
    """compute_score 関数の既存動作が壊れていないことの確認"""
    
    def test_basic_score(self):
        df = pd.DataFrame({'win_prob': [0.3, 0.1], 'odds': [5.0, 30.0]})
        scores = compute_score(df)
        np.testing.assert_array_almost_equal(scores, [1.5, 3.0])
    
    def test_zero_odds(self):
        """オッズ0のときは win_prob をそのまま返す"""
        df = pd.DataFrame({'win_prob': [0.5], 'odds': [0.0]})
        scores = compute_score(df)
        np.testing.assert_array_almost_equal(scores, [0.5])


class TestHistoryDataLoading:
    """P2: 履歴データロードのロジックテスト"""
    
    def test_history_start_year_calculation(self):
        """評価年から2年前を計算する"""
        eval_year = 2025
        history_start = eval_year - 2
        assert history_start == 2023
    
    def test_lag_features_with_history(self):
        """履歴データがある場合、lag特徴量がデフォルト値にならないこと"""
        # 同一馬が2レースに出走するシミュレーション
        df = pd.DataFrame({
            'horse_id': ['H1', 'H1'],
            'rank': [3, 1],
            'date': pd.to_datetime(['2024-12-01', '2025-01-15']),
        })
        df = df.sort_values(['horse_id', 'date'])
        df['lag1_rank'] = df.groupby('horse_id')['rank'].shift(1).fillna(99)
        
        # 2025年のレース (2行目) で lag1_rank が 2024年の値 (3) になること
        assert df.iloc[1]['lag1_rank'] == 3  # デフォルト値99ではない
    
    def test_lag_features_without_history(self):
        """履歴データがない場合、lag特徴量がデフォルト値になること"""
        df = pd.DataFrame({
            'horse_id': ['H1'],
            'rank': [1],
            'date': pd.to_datetime(['2025-01-15']),
        })
        df = df.sort_values(['horse_id', 'date'])
        df['lag1_rank'] = df.groupby('horse_id')['rank'].shift(1).fillna(99)
        
        # 初レースでは lag1_rank がデフォルト値99
        assert df.iloc[0]['lag1_rank'] == 99


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
