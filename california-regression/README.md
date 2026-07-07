California Housing 線形回帰モデル比較

4種類の線形回帰モデル（Normal / Ridge / LASSO / Elastic Net）を使って、
California Housing データセット（住宅価格の予測）を解析する。

課題の狙い

4モデルの違いは「重み係数へのペナルティ（正則化）」の付け方にある。

モデル正則化特徴Normalなし純粋に平均二乗誤差を最小化RidgeL2（Σw²）係数を全体的に小さく縮める。全特徴量が残るLASSOL1（Σ|w|）一部の係数をちょうど0にする（特徴量選択）Elastic NetL1 + L2両者のブレンド

この違いを実データで観察するのが目的。

セットアップ

bash# 仮想環境（任意だが推奨）
python -m venv venv
source venv/bin/activate        # Windows は venv\Scripts\activate

# 依存パッケージ
pip install -r requirements.txt

実行

bashpython main.py

実行すると以下が出力される。


データの概要（サンプル数・特徴量・スケールの違い・相関）
交差検証で選ばれた各モデルの正則化の強さ（alpha）
テストデータでの予測精度（RMSE・R²）
モデルごとの係数比較（LASSO / Elastic Net でいくつ0になったか）
figures/ に2枚のグラフ（RMSE比較・係数比較）


ディレクトリ構成

california_regression/
├── main.py              # 全体を通す実行スクリプト
├── requirements.txt
├── README.md
└── src/
    ├── data_loader.py   # データ読み込みと概要表示
    ├── preprocess.py    # 訓練/テスト分割と標準化
    ├── models.py        # 4モデルの定義と alpha 自動選択
    └── evaluate.py      # 評価（RMSE, R²）と可視化

メモ


訓練・テストは 8:2 で分割している。
正則化の前に標準化（StandardScaler）を行う。これは特徴量ごとの
スケール差により L1/L2 ペナルティが不公平になるのを防ぐため。
標準化のパラメータは訓練データのみから計算し、データ漏洩を避けている。