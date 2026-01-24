# 🏇 Keiba Prediction App (競馬予想アプリ)

Netkeibaの出馬表URLを入力として、機械学習モデルに基づいた高精度な予想を行うPythonデスクトップアプリケーションです。
LightGBMを用いた多クラス分類モデルにより、単なる的中率だけでなく回収率を意識した「スコア」を算出します。

## ✨ 特徴

- **Advanced ML Strategy**:
    - **アルゴリズム**: LightGBM (Gradient Boosting Decision Tree)
    - **特徴量**:
        - **Speed Index**: コース・距離ごとの標準タイム偏差値（スピード指数）。
        - **Target Encoding**: 騎手の勝率データを活用。
        - **Context Features**: コース適性、天候、距離、馬場状態を考慮。
        - **Lag Features (過去走)**: 前走の着順や指数、出走間隔を推論時に動的に参照。
    - **ベッティング戦略**: 「確率の4乗 × オッズ」によるスコアリングで、期待値の高い馬を選抜。

- **Backtesting & Verification**:
    - `betting.yaml` による条件指定（競馬場・レース番号）が可能なバックテスト機能。
    
- **Multi-Mode Prediction**:
    - **Search Mode**: 日付・競馬場を指定してレースを検索・予想。
    - **Direct URL**: 出馬表URLを直接入力して予想。

- **モダンなGUI**: Windows 11ライクな洗練されたデザイン (`customtkinter`)。
- **学習パイプライン**: 過去10年分（2016-2025）のデータ学習に対応。

## 🛠️ 必要要件

- Windows OS
- Python 3.10+
- Internet Connection

## 🚀 インストール方法

1. **リポジトリのクローン**
   ```bash
   git clone https://github.com/lewon810/keiba.git
   cd keiba
   ```

2. **仮想環境の作成と有効化**
   ```powershell
   python -m venv venv
   .\venv\Scripts\Activate.ps1
   ```

3. **依存ライブラリのインストール**
   ```bash
   pip install -r requirements.txt
   ```

## 🖥️ 使い方

### 1. アプリケーションで予想する (UI)
```bash
python -m app.main
```
- **Search Mode**: 日にちを指定し「Search & Predict」を押すと、その日の全レースを予想します。
- **Direct URL**: NetkeibaのURLを直接貼り付けて予想します。

### 2. モデル精度を検証する (Evaluation)
`betting.yml` で条件（対象競馬場など）や**賭け式**を設定し、バックテストを実行します。

```yaml
# betting.yml example
target_places: ["06"] # Nakayama
betting_type: win     # win, place, trifecta, box_trifecta, uma_ren, wide
```

```powershell
# 例: 2025年のデータでバックテスト
python -m train.evaluate --start 2025 --end 2025
```
- **win**: 単勝 (ROI算出あり)
- **place**: 複勝 (的中率のみ)
- **trifecta**: 3連単 (1-2-3着完全一致)
- **box_trifecta**: 3連単ボックス (上位3頭が1-3着入線)

### 3. モデルの再学習 (Training)
最新データを取り込んでモデルを再学習させる手順です。

#### Step 1: データの収集 (Scraping)
Netkeibaから過去のレース結果を収集します。（時間がかかります）
```powershell
# 例: 2023年〜2024年のデータを収集
python -m train.scraper_bulk --start 2023 --end 2024
```
※ `train/data/raw/` にCSVファイルが保存されます。

#### Step 2: モデルの学習 (Training)
収集したデータを使ってLightGBMモデルを学習させます。
```powershell
# 自動的にrawデータを読み込み、前処理・学習・保存を行います
python -m train.train
```
※ 学習済みモデルは `train/models/lgbm_ranker_v2.pkl` に保存されます。

## 📂 プロジェクト構成

```
keiba/
├── app/                  # 予測アプリケーション (Inference)
│   ├── main.py           # GUI
│   ├── scraper.py        # スクレイパー (Search機能含む)
│   ├── predictor.py      # ML推論エンジン
│   └── history_loader.py # 過去走データ参照モジュール
├── train/                # 機械学習パイプライン (Training)
│   ├── scraper_bulk.py   # 過去ログ収集スクレイパー
│   ├── evaluate.py       # 精度検証・バックテスト
│   └── data/             # データ置き場
├── betting.yaml          # 検証用設定ファイル (New!)
└── requirements.txt      
```

## ⚠️ 注意事項

- 本アプリケーションは学習・研究目的で作成されています。
- 予想結果は的中を保証するものではありません。

## 📄 License

MIT License
