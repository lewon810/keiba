import os

# Base Directories
# Base Directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) # train/
DATA_DIR = os.path.join(BASE_DIR, 'data')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
MODEL_DIR = os.path.join(BASE_DIR, 'models') # train/models
MODEL_PATH = os.path.join(MODEL_DIR, 'lgbm_ranker_v2.pkl')

# Feature Engineering Settings
CATEGORY_COLS = ['jockey_id', 'horse_id', 'trainer_id', 'course_type', 'weather', 'condition']
NUM_CLASSES = 1 # Ranker output is 1D score (previously 4 for classification)

# Prediction Settings
POWER_EXPONENT = 4 # Default exponent for Score = P^n * Odds
