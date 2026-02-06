# System Requirements

## Overview
The application is a horse racing prediction system consisting of two main components: `app/` and `train/`.


## Directory Structure
```
keiba/
├── docs/               # Documentation & Requirements
├── app/                # Main App (Inference & UI)
├── train/              # ML Logic & Training Scripts
│   ├── scraper.py      # Bulk scraper for training data
│   ├── preprocess.py   # Feature engineering pipeline
│   ├── train.py        # LightGBM training script
│   └── data/           # Raw and processed data storage
└── README.md
```

## Detailed ML Requirements

### 1. Data Collection
- Source: Netkeiba.com
- Scope: Historical data (e.g., past 10 years).
- Tables: Race Info, Race Results, Horse Entries, Horse Details.

### 2. Feature Engineering
- **Target Variable**: `rank_class` (0: 1st, 1: 2-3rd, 2: 4-5th, 3: 6th+).
- **Key Features**:
    - **Speed Index**: Standardized time based on course/distance averages.
    - **Target Encoding**: Jockey win rates, Sire mud index (calculated using past data only to avoid leakage).
    - **Lag Features**: Previous race result, time, interval days, distance change.
    - **Categorical**: Horse ID, Jockey ID, Trainer ID (Label Encoded).

### 3. Model Training
- **Library**: `lightgbm`
- **Objective**: `multiclass` (num_class=4)
- **Metric**: `multi_logloss`
- **Validation**: Time-series split (no random shuffle).

### 4. Betting Strategy (Inference)
- **Score Calculation**: $Score = (Probability_{Win})^4 \times Odds$
- **Selection**: Filter horses with high scores as candidates.
