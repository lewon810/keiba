---
trigger: always_on
---

## Rule
- チャットには日本語で応答してください。
- 原則コメントも日本語で書いてください。
- python実行時はvenv上で実行することを忘れないでください。
- 実行しているOSを確認し、Windowsの場合はPowerShellを利用してください。
- PowerShellで出力したファイルの文字コードは常にUTF-8になるように注意してください。
- コードに修正を加える際、README.mdの更新が必要であれば日本語で更新してください。
- 学習データを増やした際は全ての予想・学習プログラムで特徴量が一致するようにしてください。
- ディレクトリやファイル名を変更した場合はREADME更新とtestがPASSすることを確認してください。
- コードに修正を加える際、以下のソースコードの動作を保証するようにtestを作成・実行・更新してください。
    - `train/evaluate_model.py`
    - `train/evaluate.py`
    - `train/preprocess.py`
    - `train/train.py`
    - `app/scraper.py`
    - `app/predictor.py`
    - `app/main.py`  

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

## Architecture

### Learn (Machine Learning) - `train/`
- **Role**: Model training and data construction.
- **Functionality**:
    - Implements machine learning logic based on the theory described in `docs/strategy.html`.
    - **Algorithm**: LightGBM (Multi-class classification: Winner, Placed, Board, Lost).
    - **Features**: Speed Index (standardized), Target Encoding (Jockey/Sire), Lag features (past performance), Contextual features.
    - **Strategy**: "Power of 4" (Score = $P^4 \times Odds$).
    - **Data Source**: Netkeiba historical data (bulk scraping required).
    - **Validation**: Time-series split (Train -> Valid -> Test).
    - constructs training data.
    - Outputs a trained model or reference data for the `prediction` component.
- **Theory Source**: `docs/theory.html`