"""
California Housing データセットの読み込みと概要確認。

sklearn に内蔵されている fetch_california_housing を使う。
1990年カリフォルニア国勢調査に基づく地区(block group)単位のデータで、
目的変数は住宅価格の中央値(単位: 10万ドル)。
"""

import pandas as pd
from sklearn.datasets import fetch_california_housing


def load_data():
    """California Housing を pandas の形で返す。

    まず sklearn 内蔵の fetch_california_housing を試す。
    ネットワーク等の都合で取得できない場合は OpenML 経由での取得を試みる。

    Returns:
        X (pd.DataFrame): 8個の説明変数
        y (pd.Series): 目的変数(住宅価格の中央値, 単位10万ドル)
    """
    try:
        dataset = fetch_california_housing(as_frame=True)
        X = dataset.data
        y = dataset.target
    except Exception:
        # フォールバック: OpenML から取得
        from sklearn.datasets import fetch_openml
        dataset = fetch_openml(name="california_housing", version=1, as_frame=True)
        X = dataset.data
        y = dataset.target
    return X, y


def describe_data(X, y):
    """データの概要を標準出力に表示する。"""
    print("=" * 60)
    print("California Housing データセットの概要")
    print("=" * 60)
    print(f"サンプル数: {X.shape[0]}")
    print(f"特徴量の数: {X.shape[1]}")
    print()
    print("特徴量の一覧:")
    for col in X.columns:
        print(f"  - {col}")
    print()
    print("特徴量の統計量(スケールの違いに注目):")
    print(X.describe().T[["mean", "std", "min", "max"]])
    print()
    print(f"目的変数 y の範囲: {y.min():.2f} 〜 {y.max():.2f} (単位: 10万ドル)")
    print()
    print("特徴量どうしの相関(絶対値が大きいペアほど正則化の効果が出やすい):")
    print(X.corr().round(2))


if __name__ == "__main__":
    X, y = load_data()
    describe_data(X, y)