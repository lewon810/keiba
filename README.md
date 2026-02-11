# 🏇 Keiba Prediction App (競馬予想アプリ)

Netkeibaの出馬表URLを入力として、機械学習モデルに基づいた高精度な予想を行うPythonデスクトップアプリケーションです。
LightGBMを用いた多クラス分類モデルにより、単なる的中率だけでなく回収率を意識した「スコア」を算出します。
https://lewon810.github.io/keiba/

## ✨ 特徴

- **Advanced ML Strategy**:
    - **アルゴリズム**: LightGBM (Gradient Boosting Decision Tree)
    - **特徴量**:
        - **Speed Index**: コース・距離ごとの標準タイム偏差値（スピード指数）。
        - **Pace (ペース)**: 逃げ・先行（1コーナー2位以内）の頭数と比率。レース展開を予測。
        - **Pedigree (血統)**: 種牡馬および母父の勝率データを活用。
        - **Target Encoding**: 騎手・調教師の勝率データを活用。
        - **Context Features**: コース適性、天候、距離、馬場状態を考慮。
        - **Lag Features (過去走)**: 前走の着順や指数、**上がり3F**、出走間隔を推論時に動的に参照。
    - **ベッティング戦略**: 「確率のn乗 × オッズ」によるスコアリング。
        - パラメータ `power` (デフォルト4) により穴馬への感度を調整可能。

- **Backtesting & Verification**:
    - `train/evaluate_settings.yml` による詳細な条件設定。
    - **Range Evaluation**: 異なる指数（Power）やレース番号を範囲指定して一括検証。
    - **HTML Reports**: グラフや詳細な的中履歴を含むレポートを自動生成。

- **Multi-Mode Prediction**:
    - **Search Mode**: 日付・競馬場を指定してレースを検索・予想。
    - **Direct URL**: 出馬表URLを直接入力して予想。

- **モダンなGUI**: Windows 11ライクな洗練されたデザイン (`customtkinter`)。
- **学習パイプライン**: 過去10年分（2016-2025）のデータ学習に対応。

- **CI/CD Integration**:
    - **GitHub Pages**: 予測結果および検証レポートを自動デプロイ。
    - **Automated Scraping**: GitHub Actionsによる血統データの自動収集機能。

## 🛠️ 必要要件

- Windows OS (GUI) / Linux (CI)
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

**CLI - Batch Prediction & Reports:**
```bash
# 特定の日付の予想レポート(HTML)を生成
python -m app.report.predict_html_generator --date 20250125
```

### 2. モデル精度を検証する (Evaluation)
`train/evaluate_settings.yml` で条件を設定し、評価用HTMLレポートを生成します。

```powershell
# 指定期間（例:2025年）のデータで検証レポートを生成
python -m train.report.evaluate_html_generator --start 2025 --end 2025
```

### 3. データ収集とモデル学習

#### Step 1: レース結果の収集
```powershell
python -m train.scraper_bulk --start 2016 --end 2025
```
※ `train/data/raw/` にCSVファイルが保存されます。

#### Step 2: 血統情報の収集 (Pedigree)
GitHub Actionsで実行するか、ローカルで実行します。
```powershell
python -m train.scraper_horse
```
※ `train/data/raw/horse_profiles.csv` に保存されます。

#### Step 3: モデルの学習
```powershell
# 自動的にrawデータを読み込み、前処理・学習・保存を行います
python -m train.train
```
※ 学習済みモデルは `train/models/lgbm_ranker_v2.pkl` に保存されます。

## 📂 プロジェクト構成

```
keiba/
├── app/                  # 予測・推論レイヤー
│   ├── main.py           # GUIアプリケーション
│   ├── scraper.py        # スクレイパー (Search機能含む)
│   ├── predictor.py      # 推論エンジン (LightGBM)
│   └── report/           # 予測レポート生成
├── train/                # 学習パイプライン
│   ├── scraper_bulk.py   # レース結果収集スクレイパー
│   ├── scraper_horse.py  # 血統情報収集スクレイパー [NEW]
│   ├── preprocess.py     # 特徴量エンジニアリング (血統統合済み)
│   ├── train.py          # モデル学習
│   ├── evaluate.py       # 検証エンジン
│   ├── evaluate_settings.yml # 検証用設定ファイル
│   └── report/           # 評価レポート生成
├── .github/workflows/    # CI/CD設定 (Scraping, Deploy)
├── data/                 # 学習データ・モデル (Git管理下)
└── requirements.txt      
```

## ⚠️ 注意事項

- 本アプリケーションは学習・研究目的で作成されています。
- 予想結果は的中を保証するものではありません。

## 📄 License

MIT License
