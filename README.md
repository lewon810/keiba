# 🏇 Keiba Prediction App (競馬予想アプリ)

Netkeibaの出馬表URLを入力として、競走馬の予想を行うPythonデスクトップアプリケーションです。
モダンなGUI (`customtkinter`) を採用し、直感的な操作が可能です。

## ✨ 特徴

- **モダンなGUI**: Windows 11ライクな洗練されたデザイン。
- **自動スクレイピング**: Netkeibaの出馬表ページから馬名、騎手、馬番などのデータを自動取得。
- **予想ロジック**: 取得したデータに基づき、独自のロジック（現在はサンプル実装として馬名シードによるランダム予測）で予想ランキングを表示。
- **高速動作**: 軽量なスクレイピングとマルチスレッド処理による非同期実行。

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

1. **アプリケーションの起動**
   ```bash
   python src/main.py
   ```

2. **URLの入力**
   - Netkeibaの出馬表ページ（例: `https://race.netkeiba.com/race/shutuba.html?race_id=...`）のURLをコピーします。
   - アプリのテキストボックスに貼り付けます。

3. **予想開始**
   - 「Predict」ボタンをクリックします。
   - 下部のログエリアに取得データと予想ランキングが表示されます。

## 📂 プロジェクト構成

```
keiba/
├── src/
│   ├── main.py           # GUIアプリケーションのエントリーポイント
│   ├── scraper.py        # スクレイピング処理 (BeautifulSoup4)
│   ├── predictor.py      # 予想ロジック
│   └── integration_test.py # 統合テスト用スクリプト
├── requirements.txt      # 依存ライブラリ一覧
├── README.md             # ドキュメント
└── .gitignore            # Git除外設定
```

## ⚠️ 注意事項

- 本アプリケーションは学習・研究目的で作成されています。
- スクレイピングを行う際は、対象サイトの利用規約を遵守し、サーバーに過度な負荷をかけないようご注意ください。
- 予想結果は的中を保証するものではありません。

## 📄 License

MIT License
