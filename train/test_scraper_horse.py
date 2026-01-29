import unittest
import pandas as pd
import os
import shutil
from train import scraper_horse

class TestScraperHorse(unittest.TestCase):
    def setUp(self):
        self.test_dir = "train/test_data"
        os.makedirs(self.test_dir, exist_ok=True)
        self.target_path = os.path.join(self.test_dir, "target.csv")
        self.source_path = os.path.join(self.test_dir, "source.csv")

    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_merge_profiles_update(self):
        # Create target (Simulate existing DB)
        # ID 1: Old Name
        # ID 2: Existing Horse
        df_target = pd.DataFrame({
            "horse_id": ["1", "2"],
            "name": ["OldName", "ExistingHorse"]
        })
        df_target.to_csv(self.target_path, index=False)

        # Create source (Simulate Scraped Data)
        # ID 1: New Name (Update)
        # ID 3: New Horse (Insert)
        df_source = pd.DataFrame({
            "horse_id": ["1", "3"],
            "name": ["NewName", "NewHorse"]
        })
        df_source.to_csv(self.source_path, index=False)

        # Execute Merge
        # We expect scraper_horse.merge_profiles to merge source into target
        # And ideally UPDATE existing records (ID 1 -> NewName) and INSERT new records (ID 3)
        scraper_horse.merge_profiles(self.source_path, self.target_path)

        # Verify Result
        df_result = pd.read_csv(self.target_path)
        
        # Sort by ID for consistent checking
        df_result = df_result.sort_values("horse_id").reset_index(drop=True)
        
        # Check ID 1
        row1 = df_result[df_result["horse_id"] == 1].iloc[0]
        # CURRENT BEHAVIOR (FIXED): "NewName" because keep='last' and source is last
        self.assertEqual(row1["name"], "NewName") 
        
        # Check ID 2 (Should remain)
        row2 = df_result[df_result["horse_id"] == 2].iloc[0]
        self.assertEqual(row2["name"], "ExistingHorse")

        # Check ID 3 (Should be added)
        row3 = df_result[df_result["horse_id"] == 3].iloc[0]
        self.assertEqual(row3["name"], "NewHorse")

        print("Test Result Data:")
        print(df_result)

if __name__ == "__main__":
    unittest.main()
