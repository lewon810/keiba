# 🏇 Keiba Prediction System (競馬予想システム)

GitHub Actionsを活用した自動競馬予想システムです。LightGBMを用いた機械学習モデルにより、単なる的中率だけでなく回収率を意識した「スコア」を算出し、予測結果をGitHub Pagesで公開します。
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

- **自動化パイプライン**: 過去10年分（2016-2025）のデータ学習に対応。

- **CI/CD Integration**:
    - **GitHub Pages**: 予測結果および検証レポートを自動デプロイ。
    - **Automated Scraping**: GitHub Actionsによる血統データの自動収集機能。

## 🛠️ 必要要件

- Python 3.10+
- Internet Connection
- GitHub Account (GitHub Actions / Pages使用)

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

### 1. GitHub Actionsで自動予測

このシステムは主にGitHub Actionsで動作します：

- **Predict Workflow** (`predict.yml`): 3時間ごとに自動実行され、今週のレースを予測
- **Evaluate Workflow** (`evaluate.yml`): 毎週月曜日にモデルの精度を評価
- **Deploy Workflow** (`deploy.yml`): 毎日予測結果と評価レポートをGitHub Pagesにデプロイ

**手動実行**: リポジトリの「Actions」タブから各ワークフローを手動でトリガー可能です。

### 2. ローカルで予測レポートを生成

```bash
# 特定の日付の予想レポート(HTML)を生成
python -m app.report.predict_html_generator --date 20250125
```

### 3. モデル精度を検証する (Evaluation)

```powershell
# 指定期間（例:2025年）のデータで検証レポートを生成
python -m train.report.evaluate_html_generator --start 2025 --end 2025
```

### 4. データ収集とモデル学習

#### GitHub Actionsで自動実行（推奨）

- **Scrape Race Results** (`scrape-race-results.yml`): レース結果を年単位で収集
- **Scrape Horse Profiles** (`scrape-horse.yml`): 血統情報を自動収集

これらのワークフローは「Actions」タブから手動で実行できます。

#### ローカルで実行する場合

**Step 1: レース結果の収集**
```powershell
python -m train.scraper_bulk --start 2016 --end 2025
```
※ `train/data/raw/` にCSVファイルが保存されます。

**Step 2: 血統情報の収集**
```powershell
python -m train.scraper_horse
```
※ `train/data/raw/horse_profiles.csv` に保存されます。

**Step 3: モデルの学習**
```powershell
python -m train.train
```
※ 学習済みモデルは `train/models/lgbm_ranker_v2.pkl` に保存されます。

## 📂 プロジェクト構成

```
keiba/
├── .github/workflows/    # GitHub Actions ワークフロー
│   ├── predict.yml       # 予測実行（3時間ごと）
│   ├── evaluate.yml      # モデル評価（週次）
│   ├── deploy.yml        # GitHub Pagesデプロイ（日次）
│   ├── scrape-race-results.yml  # レース結果収集
│   └── scrape-horse.yml  # 血統情報収集
├── app/                  # 予測・推論レイヤー
│   ├── scraper.py        # スクレイパー (レース検索機能)
│   ├── predictor.py      # 推論エンジン (LightGBM)
│   ├── history_loader.py # 履歴データローダー
│   └── report/           # 予測レポート生成
├── train/                # 学習パイプライン
│   ├── scraper_bulk.py   # レース結果収集スクレイパー
│   ├── scraper_horse.py  # 血統情報収集スクレイパー
│   ├── preprocess.py     # 特徴量エンジニアリング
│   ├── train.py          # モデル学習
│   └── report/           # 評価レポート生成
├── deploy/               # デプロイスクリプト
│   └── index_generator.py # GitHub Pagesインデックス生成
├── data/                 # 学習データ・モデル (Git管理下)
└── requirements.txt      
```

## ⚠️ 注意事項

- 本アプリケーションは学習・研究目的で作成されています。
- 予想結果は的中を保証するものではありません。

## 📄 License

MIT License
