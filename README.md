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
    - 未学習データを用いた精度検証（Accuracy / ROIシミュレーション）機能を搭載。
    
- **モダンなGUI**: Windows 11ライクな洗練されたデザイン (`customtkinter`)。
- **自動スクレイピング**: Netkeibaの出馬表から最新オッズやレース条件を自動取得。
- **学習パイプライン**: 過去10年分（2016-2025）のデータ学習に対応し、継続的なモデル改善が可能。

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

### 1. アプリケーションで予想する (Inference)
学習済みモデル (`train/models/lgbm_ranker_v2.pkl`) と過去データ (`app/history_loader.py`) を使用して予想を行います。

```bash
python app/main.py
```
- Netkeibaの出馬表URLを貼り付けて「Predict」を押すだけ。

### 予想結果の見方
出力エリアには以下のような形式で予想結果が表示されます（スコアが高い順）。

```text
Prediction Ranking (Normalized Score):
Context: sunny / 2000m  (← 現在のレース条件: 天候 / 距離)
----------------------------------------
◎  1. 馬名A (Odds: 4.5, Score: 1.2345)
○  2. 馬名B (Odds: 2.1, Score: 0.9876)
▲  3. 馬名C (Odds: ---.-, Score: 0.1234)
...
```

- **◎ ○ ▲ △**: 予測スコアの上位4頭に付与される印です。
- **Odds**: Netkeibaから取得したオッズです（取得できない場合は `---.-`）。
- **Score (Normalized Score)**: AIモデルが算出した「勝ちやすさ」を0から1の範囲に正規化した指標です。
    - 数値が大きいほど、モデルが「勝つ可能性が高い」と評価しています。
    - レース内での相対評価であり、0.0が最も評価が低く、1.0が最も評価が高い馬となります（全馬同じ評価の場合は0.5）。

### 2. オリジナルモデルを学習する (Training)
過去のデータを使ってモデルを再学習させることができます。

1. **データ収集 (スクレイピング)**
   ```powershell
   # 例: 2016年から2025年の全データを取得 (数時間かかります)
   python -m train.scraper_bulk --start 2016 --end 2025
   ```

2. **モデル学習**
   ```powershell
   python -m train.train
   ```
   - Features: Speed Index, Metadata, Lag Features等を自動生成して学習します。

### 3. モデル精度を検証する (Evaluation)
集めたデータの一部を「未知のデータ」として扱い、モデルの実力を検証します。

```powershell
# 例: 2025年のデータでバックテスト
python -m train.evaluate --start 2025 --end 2025
# 既存のCSVを指定してバックテスト
python -m train.evaluate --csv path/to/results.csv
```
- 正解率 (Accuracy) や、簡単なROI（回収率）シミュレーション結果が表示されます。

## 📂 プロジェクト構成

```
keiba/
├── app/                  # 予測アプリケーション (Inference)
│   ├── main.py           # GUI
│   ├── scraper.py        # 出馬表スクレイパー & メタデータ抽出
│   ├── predictor.py      # ML推論エンジン
│   └── history_loader.py # 過去走データ参照モジュール
├── train/                # 機械学習パイプライン (Training)
│   ├── scraper_bulk.py   # 過去ログ収集スクレイパー
│   ├── preprocess.py     # 特徴量エンジニアリング (Speed Index等)
│   ├── train.py          # LightGBM学習スクリプト
│   ├── evaluate.py       # 精度検証・バックテスト
│   └── data/             # データ置き場
├── docs/                 # 要件定義書など
└── requirements.txt      
```

## ⚠️ 注意事項

- 本アプリケーションは学習・研究目的で作成されています。
- スクレイピングを行う際は、サーバーに過度な負荷をかけないようご注意ください（スクリプト内には自動的なWait処理が含まれています）。
- 予想結果は的中を保証するものではありません。

## 📄 License

MIT License
