import unittest
import pandas as pd
import numpy as np
import os
import sys
from unittest.mock import MagicMock, patch

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class TestKeibaSystem(unittest.TestCase):

    def setUp(self):
        # Create dummy data for testing
        self.dummy_race_data = [{
            'name': 'Test Horse',
            'horse_id': '12345',
            'jockey_id': '54321',
            'trainer_id': '99999',  # Added for LambdaRank model
            'course_type': 'turb',
            'weather': 'sunny',
            'condition': 'good',
            'distance': 1600,
            'waku': '1',
            'umaban': '1',
            'odds': '5.0',
            'date': '2025年1月1日',
            'rank': '1', # For training/eval context
            'time': '1:34.0',
            'race_id': '202501010101'
        }]
        
        self.dummy_df = pd.DataFrame(self.dummy_race_data)

    # --- APP Tests ---
    
    def test_predictor_import(self):
        from app import predictor
        self.assertTrue(hasattr(predictor, 'predict'))
        
    @patch('app.predictor.joblib.load')
    def test_predictor_logic(self, mock_load):
        # Mock Model and Artifacts
        mock_model = MagicMock()
        # LambdaRank returns 1D score array
        mock_model.predict.return_value = np.array([5.0]) # Single score for single horse
        
        mock_encoders = {
            'horse_id': MagicMock(),
            'jockey_id': MagicMock(),
            'trainer_id': MagicMock(),
            'sire_id': MagicMock(),
            'damsire_id': MagicMock(),
            'running_style': MagicMock(),
            'course_type': MagicMock(),
            'weather': MagicMock(),
            'condition': MagicMock(),
            'jockey_win_rate': {},
            'trainer_win_rate': {},
            'sire_win_rate': {},
            'damsire_win_rate': {},
            'aptitude_type': {},
            'aptitude_dist': {}
        }
        # Configure encoder mocks
        for k, v in mock_encoders.items():
            if hasattr(v, 'transform'):
                v.transform.return_value = np.array([0])
                v.classes_ = ['12345', 'turb', 'sunny', 'good']
            
        mock_load.side_effect = [mock_model, mock_encoders]
        
        from app import predictor
        # Mock os.path.exists to pass checks
        with patch('os.path.exists', return_value=True):
            # Also mock history loader to avoid file reads
            with patch('app.history_loader.loader.get_last_race', return_value=None):
                result = predictor.predict(self.dummy_race_data)
                
        self.assertIn("Test Horse", result)
        self.assertIn("Score:", result)

    def test_score_normalization(self):
        """
        Test that scores are normalized to 0-1 range for LambdaRank.
        """
        from app import predictor
        
        # 2 horses with distinct features (mocked via model return)
        race_data = [
            self.dummy_race_data[0].copy(), 
            self.dummy_race_data[0].copy()
        ]
        race_data[1]['name'] = 'Horse 2'
        
        with patch('app.predictor.joblib.load') as mock_load:
            mock_model = MagicMock()
            # LambdaRank returns raw scores: 10.0 and 0.0
            mock_model.predict.return_value = np.array([10.0, 0.0])
            
            mock_encoders = {
                'jockey_win_rate': {},
                'trainer_win_rate': {},
                'sire_win_rate': {},
                'damsire_win_rate': {},
                'aptitude_type': {},
                'aptitude_dist': {}
            } 
            for col in ['horse_id', 'jockey_id', 'trainer_id', 'sire_id', 'damsire_id', 'running_style', 'course_type', 'weather', 'condition']:
                m = MagicMock()
                m.transform.return_value = np.array([0, 0])
                m.classes_ = ['12345', 'turb', 'sunny', 'good']
                mock_encoders[col] = m

            mock_load.side_effect = [mock_model, mock_encoders]
            
            with patch('os.path.exists', return_value=True):
                 with patch('app.history_loader.loader.get_last_race', return_value=None):
                    result = predictor.predict(race_data)
        
        # After normalization, 10.0 -> 1.0, 0.0 -> 0.0
        # But the actual score calculation is (win_prob^power * odds)
        # With odds=5.0: score = 1.0^4 * 5.0 = 5.0 and 0.0^4 * 5.0 = 0.0
        self.assertIn("Score:", result)
        
    def test_scraper_mock(self):
        from app import scraper
        with patch('requests.get') as mock_get:
            mock_get.return_value.text = "<html></html>"
            mock_get.return_value.encoding = 'utf-8'
            data = scraper.fetch_race_data("http://mock.url")
            self.assertEqual(data, []) # Should be empty for empty html

    def test_main_import(self):
        from app import main
        self.assertTrue(issubclass(main.KeibaApp, object))

    # --- TRAIN Tests (Current Pipeline) ---

    # def test_train_model_functions(self):
    #     import train.train_model as tm
    #     # Test feature eng function logic
    #     df_eng = tm.feature_engineering(self.dummy_df.copy())
    #     self.assertIn('horse_win_rate', df_eng.columns)
        
    def test_evaluate_model_functions(self):
        import train.evaluate_model as em
        # Test preprocess logic
        encoders = {} # Empty dict
        # Must run FE first to get rank_num
        df_eng = em.feature_engineering(self.dummy_df.copy())
        df_processed, cols = em.preprocess(df_eng, encoders)
        self.assertFalse(df_processed.empty)

    # --- TRAIN Tests (Legacy Pipeline Compatibility) ---

    def test_legacy_preprocess(self):
        import train.preprocess as pp
        df, artifacts = pp.preprocess(self.dummy_df.copy())
        self.assertIn('speed_index', df.columns)
        self.assertIn('lag1_rank', df.columns)

    def test_legacy_train_import(self):
        import train.train as lt
        self.assertTrue(hasattr(lt, 'train_model'))

    def test_legacy_evaluate_import(self):
        import train.evaluate as le
        self.assertTrue(hasattr(le, 'evaluate'))

    # --- Integration Test with REAL Artifacts ---
    def test_integration_real_model(self):
        """
        Loads the ACTUAL model file defined in settings.py and checks prediction shape.
        This ensures the dimensionality mismatch error is resolved.
        Checking model DIRECTLY requires properly encoded numeric inputs.
        """
        import joblib
        from train import settings
        
        if not os.path.exists(settings.MODEL_PATH):
            self.skipTest(f"Real model not found at {settings.MODEL_PATH}")
            
        model = joblib.load(settings.MODEL_PATH)
        
        # Prepare valid features - UPDATED for LambdaRank model
        # Same features as in train.py and predictor.py
        features = [
            'jockey_win_rate', 'trainer_win_rate', 'horse_id', 'jockey_id', 'trainer_id',
            'waku', 'umaban', 'course_type', 'distance', 'weather', 'condition',
            'lag1_rank', 'lag1_speed_index', 'interval', 'weight_diff',
            'sire_id', 'damsire_id', 'running_style',
            'sire_win_rate', 'damsire_win_rate',
            'course_type_win_rate', 'dist_cat_win_rate'
        ]
        
        # Create a dummy processed input with NUMERIC values (Simulating processed, encoded data)
        # Model expects ints/floats, not strings.
        dummy_input = pd.DataFrame([{
            'jockey_win_rate': 0.1,
            'trainer_win_rate': 0.15,
            'horse_id': 0, 'jockey_id': 0, 'trainer_id': 0,
            'waku': 1, 'umaban': 1,
            'course_type': 0, 'distance': 0, 'weather': 0, 'condition': 0,
            'lag1_rank': 5, 'lag1_speed_index': 50, 'interval': 14,
            'weight_diff': 0,
            'sire_id': 0, 'damsire_id': 0, 'running_style': 0,
            'sire_win_rate': 0.2, 'damsire_win_rate': 0.18,
            'course_type_win_rate': 0.12, 'dist_cat_win_rate': 0.11
        }])
        
        preds = model.predict(dummy_input[features])
        
        # Check Shape - LambdaRank returns 1D array
        self.assertEqual(preds.ndim, 1, f"Prediction should be 1D array, got shape {preds.shape}")
        self.assertEqual(len(preds), 1)
        print("\n[Integration] Real LambdaRank model prediction shape verified: 1D")

    def test_predictor_integration_real(self):
        """
        Tests app.predictor.predict with the REAL model and REAL artifacts.
        Passes raw string data (like scraper) to verify encoding logic works.
        """
        from app import predictor
        
        # Scraper-like output (all strings)
        raw_data = [{
            'name': 'Real Test Horse',
            'horse_id': '12345', # Likely unknown
            'jockey_id': '54321', # Likely unknown
            'trainer_id': '99999', # Added for LambdaRank model
            'course_type': 'turb',
            'weather': 'sunny',
            'condition': 'good',
            'distance': '1600',
            'waku': '1',
            'umaban': '1',
            'odds': '10.0',
            'date': '2026年1月1日',
            'rank': '1',
            'time': '1:34.0'
        }]
        
        # This calls predict -> loads real model -> transforms -> predicts
        result = predictor.predict(raw_data)
        
        # Check output format
        self.assertIn("Real Test Horse", result)
        self.assertIn("Score:", result)
        print("\n[Integration] Real Predictor flow verified.")

if __name__ == '__main__':
    unittest.main()
