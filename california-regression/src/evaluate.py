"""
モデルの評価と可視化。

- 予測精度: テストデータに対する RMSE と R^2 を比較する。
- 係数の比較: モデルごとに重み係数を並べ、正則化で係数がどう変化するか
  (特に LASSO / Elastic Net でいくつの係数が0になるか)を見る。
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score


def evaluate_models(models, X_test, y_test):
    """各モデルのテスト誤差(RMSE)と決定係数(R^2)を表にして返す。"""
    rows = []
    for name, model in models.items():
        y_pred = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        rows.append({"model": name, "RMSE": rmse, "R2": r2})
    return pd.DataFrame(rows).set_index("model")


def collect_coefficients(models, feature_names):
    """各モデルの係数を1つの表にまとめて返す。"""
    coef_dict = {}
    for name, model in models.items():
        coef_dict[name] = model.coef_
    coef_df = pd.DataFrame(coef_dict, index=feature_names)
    return coef_df


def count_zero_coefficients(coef_df, tol=1e-8):
    """各モデルで係数が(ほぼ)0になった特徴量の数を返す。"""
    return (coef_df.abs() < tol).sum()


def plot_metrics(metrics_df, save_path):
    """モデル別の RMSE を棒グラフにして保存する。"""
    fig, ax = plt.subplots(figsize=(7, 4))
    metrics_df["RMSE"].plot(kind="bar", ax=ax, color="#4C72B0")
    ax.set_ylabel("RMSE (lower is better)")
    ax.set_title("Test RMSE by model")
    ax.set_xlabel("")
    plt.xticks(rotation=0)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_coefficients(coef_df, save_path):
    """モデル別の係数を並べた棒グラフにして保存する。"""
    fig, ax = plt.subplots(figsize=(10, 5))
    coef_df.plot(kind="bar", ax=ax)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Coefficient")
    ax.set_title("Coefficients by model (0 means the feature was dropped)")
    ax.set_xlabel("")
    plt.xticks(rotation=45, ha="right")
    plt.legend(title="model")
    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)