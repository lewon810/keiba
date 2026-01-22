import os

# Base Directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
MODEL_DIR = os.path.join(DATA_DIR, 'model')
MODEL_PATH = os.path.join(MODEL_DIR, 'model_lgb.pkl')

# Feature Engineering Settings
CATEGORY_COLS = ['jockey_id', 'horse_id', 'trainer_id', 'course_type', 'weather', 'condition']
NUM_CLASSES = 4 # 1st, 2-3rd, 4-5th, 6th+
