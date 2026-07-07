"""
前処理: 訓練・テストへの分割と標準化。

【なぜ標準化するのか】
Ridge, LASSO, Elastic Net はいずれも「重み係数の大きさ」にペナルティをかける。
特徴量ごとにスケール(単位)が大きく違うと、値の大きい特徴量ほど不当にペナルティを
受けてしまい、公平な比較ができない。そこで全特徴量を平均0・標準偏差1に揃える。

【データ漏洩(data leakage)を避ける】
標準化のパラメータ(平均・標準偏差)は「訓練データだけ」から計算する。
テストデータの情報を前処理段階で覗いてしまうと評価が甘くなるため、
scaler は訓練データで fit し、テストデータには transform のみを適用する。
"""

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def split_and_scale(X, y, test_size=0.2, random_state=42):
    """訓練・テストに分割し、標準化した特徴量を返す。

    Args:
        X: 説明変数
        y: 目的変数
        test_size: テストデータの割合(0.2 = 8:2 分割)
        random_state: 再現性のための乱数シード

    Returns:
        X_train_scaled, X_test_scaled, y_train, y_test, scaler
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    scaler = StandardScaler()
    # 訓練データだけで fit する(テストの情報を混ぜない)
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler