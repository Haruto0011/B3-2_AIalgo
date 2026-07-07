"""
4種類の線形回帰モデルの定義。

いずれも「y = w0 + w1*x1 + ... + wm*xm」という線形回帰の形は同じで、
違うのは重み係数 w に対する「正則化項(ペナルティ)」の付け方だけ。

- Normal(通常の線形回帰): ペナルティなし。純粋に平均二乗誤差を最小化する。
- Ridge(L2正則化):   重みの二乗和 Σw_j^2 にペナルティ。全体を小さく縮める。
                     係数は0になりにくく、全特徴量が残る。
- LASSO(L1正則化):   重みの絶対値和 Σ|w_j| にペナルティ。一部の係数を
                     ちょうど0にする → 特徴量選択の効果を持つ。
- Elastic Net:       L1 と L2 を混ぜたもの。両者の良いとこ取り。
                     l1_ratio でブレンド比率を決める。

alpha は正則化の強さ。大きいほどペナルティが強く、係数がより縮む。
"""

from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.model_selection import GridSearchCV


def build_models(alpha=1.0, l1_ratio=0.5):
    """4種類のモデルを辞書で返す。

    Args:
        alpha: 正則化の強さ(Normal 以外に適用)
        l1_ratio: Elastic Net における L1 の比率(0〜1)

    Returns:
        dict: {モデル名: 未学習のモデル}
    """
    models = {
        "Normal": LinearRegression(),
        "Ridge": Ridge(alpha=alpha),
        "LASSO": Lasso(alpha=alpha, max_iter=10000),
        "ElasticNet": ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=10000),
    }
    return models


def build_tuned_models(X_train, y_train, cv=5):
    """交差検証で正則化の強さ(alpha)を選んだ上でモデルを返す。

    正則化ありの3モデルについては、複数の alpha 候補を交差検証で試し、
    最も精度の良い alpha を自動的に選ぶ。Normal は正則化がないのでそのまま。

    Args:
        X_train: 学習用の説明変数(標準化済み)
        y_train: 学習用の目的変数
        cv: 交差検証の分割数

    Returns:
        models (dict): {モデル名: 未学習の最良モデル}
        best_params (dict): {モデル名: 選ばれた alpha 等}
    """
    alphas = [0.001, 0.01, 0.1, 1.0, 10.0]
    l1_ratios = [0.1, 0.3, 0.5, 0.7, 0.9]

    models = {"Normal": LinearRegression()}
    best_params = {"Normal": {}}

    # Ridge
    ridge_gs = GridSearchCV(
        Ridge(), {"alpha": alphas}, cv=cv, scoring="neg_root_mean_squared_error"
    )
    ridge_gs.fit(X_train, y_train)
    models["Ridge"] = Ridge(**ridge_gs.best_params_)
    best_params["Ridge"] = ridge_gs.best_params_

    # LASSO
    lasso_gs = GridSearchCV(
        Lasso(max_iter=10000), {"alpha": alphas}, cv=cv,
        scoring="neg_root_mean_squared_error"
    )
    lasso_gs.fit(X_train, y_train)
    models["LASSO"] = Lasso(max_iter=10000, **lasso_gs.best_params_)
    best_params["LASSO"] = lasso_gs.best_params_

    # Elastic Net
    enet_gs = GridSearchCV(
        ElasticNet(max_iter=10000),
        {"alpha": alphas, "l1_ratio": l1_ratios},
        cv=cv, scoring="neg_root_mean_squared_error"
    )
    enet_gs.fit(X_train, y_train)
    models["ElasticNet"] = ElasticNet(max_iter=10000, **enet_gs.best_params_)
    best_params["ElasticNet"] = enet_gs.best_params_

    return models, best_params


def fit_models(models, X_train, y_train):
    """辞書内の全モデルを学習させる。

    Returns:
        dict: {モデル名: 学習済みモデル}
    """
    for name, model in models.items():
        model.fit(X_train, y_train)
    return models