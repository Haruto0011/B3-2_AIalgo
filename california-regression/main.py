"""
California Housing データを4種類の線形回帰モデルで解析するメインスクリプト。

実行するとデータ概要・テスト誤差・係数比較を標準出力に表示し、
figures/ 以下に2枚のグラフ(RMSE比較・係数比較)を保存する。

使い方:
    python main.py
"""

import os

from src.data_loader import load_data, describe_data
from src.preprocess import split_and_scale
from src.models import build_tuned_models, fit_models
from src.evaluate import (
    evaluate_models,
    collect_coefficients,
    count_zero_coefficients,
    plot_metrics,
    plot_coefficients,
)


def main():
    # 出力先
    fig_dir = "figures"
    os.makedirs(fig_dir, exist_ok=True)

    # 1. データ読み込みと概要確認
    X, y = load_data()
    describe_data(X, y)

    # 2. 分割(8:2)と標準化
    X_train, X_test, y_train, y_test, _ = split_and_scale(
        X, y, test_size=0.2, random_state=42
    )

    # 3. 交差検証で正則化の強さを選びつつ、4モデルを構築して学習
    models, best_params = build_tuned_models(X_train, y_train, cv=5)
    models = fit_models(models, X_train, y_train)

    print("\n" + "=" * 60)
    print("交差検証で選ばれた正則化の強さ(alpha 等)")
    print("=" * 60)
    for name, params in best_params.items():
        print(f"  {name}: {params if params else '(正則化なし)'}")

    # 4. 評価
    print("\n" + "=" * 60)
    print("テストデータでの予測精度(RMSEは小さいほど良い / R2は1に近いほど良い)")
    print("=" * 60)
    metrics_df = evaluate_models(models, X_test, y_test)
    print(metrics_df.round(4))

    # 5. 係数の比較
    print("\n" + "=" * 60)
    print("モデルごとの係数(0に近いほど、その特徴量は使われていない)")
    print("=" * 60)
    coef_df = collect_coefficients(models, X.columns)
    print(coef_df.round(4))

    print("\n各モデルで係数が0になった特徴量の数:")
    print(count_zero_coefficients(coef_df))

    # 6. 可視化
    plot_metrics(metrics_df, os.path.join(fig_dir, "rmse_by_model.png"))
    plot_coefficients(coef_df, os.path.join(fig_dir, "coefficients_by_model.png"))
    print(f"\nグラフを {fig_dir}/ に保存しました。")


if __name__ == "__main__":
    main()