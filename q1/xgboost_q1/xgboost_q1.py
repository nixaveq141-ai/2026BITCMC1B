from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, RandomizedSearchCV, train_test_split
from xgboost import XGBRegressor


SCRIPT_DIR = Path(__file__).resolve().parent
Q1_DIR = SCRIPT_DIR.parent
ROOT_DIR = Q1_DIR.parent
GRA_OUTPUT_DIR = Q1_DIR / "gra_q1" / "outputs"
OUTPUT_DIR = SCRIPT_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial"]
plt.rcParams["axes.unicode_minus"] = False
sns.set_theme(style="whitegrid")

ALL_FEATURES = [
    "ASM_A_MeanT",
    "ASD_A_SDevT",
    "BSM_B_MeanT",
    "BSD_B_SDevT",
    "CSM_C_MeanT",
    "CSD_C_SDevT",
    "ISM_I_MeanT",
    "ISD_I_SDevT",
    "ALM_A_MeanT",
    "ALD_A_SDevT",
    "BLM_B_MeanT",
    "BLD_B_SDevT",
    "CLM_C_MeanT",
    "CLD_C_SDevT",
    "ILM_I_MeanT",
    "ILD_I_SDevT",
]
TARGETS = ["DFlow", "DGap"]
RANDOM_STATE = 42


def find_preprocessed_file() -> Path:
    candidates = [
        ROOT_DIR / "预处理后数据_未标准化.csv",
        ROOT_DIR / "数据预处理" / "output" / "预处理后数据_未标准化.csv",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError("未找到预处理后数据_未标准化.csv")


def load_gra_feature_sets() -> dict[str, list[str]]:
    feature_sets: dict[str, list[str]] = {}
    for target in TARGETS:
        result_file = GRA_OUTPUT_DIR / f"{target}_gra_results.csv"
        result_df = pd.read_csv(result_file, encoding="utf-8-sig")
        feature_sets[target] = result_df["feature"].head(5).tolist()
    return feature_sets


def build_search() -> RandomizedSearchCV:
    estimator = XGBRegressor(
        objective="reg:squarederror",
        random_state=RANDOM_STATE,
        eval_metric="rmse",
        n_jobs=1,
    )
    param_distributions = {
        "n_estimators": [60, 100, 150, 220],
        "max_depth": [2, 3, 4, 5],
        "learning_rate": [0.03, 0.05, 0.08, 0.12],
        "subsample": [0.7, 0.85, 1.0],
        "colsample_bytree": [0.7, 0.85, 1.0],
        "reg_lambda": [0.5, 1.0, 3.0, 5.0],
        "gamma": [0.0, 0.05, 0.1, 0.2],
        "min_child_weight": [1, 2, 4],
    }
    n_splits = 5
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    return RandomizedSearchCV(
        estimator=estimator,
        param_distributions=param_distributions,
        n_iter=16,
        scoring="neg_root_mean_squared_error",
        n_jobs=1,
        cv=cv,
        random_state=RANDOM_STATE,
        refit=True,
    )


def compute_metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def plot_actual_vs_pred(pred_df: pd.DataFrame, target: str, model_name: str) -> None:
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=pred_df, x="actual", y="predicted", hue="dataset", s=55)
    limits = [min(pred_df["actual"].min(), pred_df["predicted"].min()), max(pred_df["actual"].max(), pred_df["predicted"].max())]
    plt.plot(limits, limits, linestyle="--", color="black", linewidth=1)
    plt.xlabel("Actual Value")
    plt.ylabel("Predicted Value")
    plt.title(f"{target} {model_name} Actual vs Predicted")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"{target}_{model_name}_actual_vs_pred.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_feature_importance(model: XGBRegressor, features: list[str], target: str, model_name: str) -> None:
    importance_df = pd.DataFrame(
        {"feature": features, "importance": model.feature_importances_}
    ).sort_values("importance", ascending=True)
    plt.figure(figsize=(8, 6))
    plt.barh(importance_df["feature"], importance_df["importance"], color="#3b82f6")
    plt.xlabel("Feature Importance")
    plt.ylabel("Feature")
    plt.title(f"{target} {model_name} Feature Importance")
    for idx, value in enumerate(importance_df["importance"]):
        plt.text(value + 0.002, idx, f"{value:.3f}", va="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"{target}_{model_name}_feature_importance.png", dpi=300, bbox_inches="tight")
    plt.close()

    importance_df.to_csv(
        OUTPUT_DIR / f"{target}_{model_name}_feature_importance.csv",
        index=False,
        encoding="utf-8-sig",
    )


def plot_metric_comparison(metrics_df: pd.DataFrame, target: str) -> None:
    target_df = metrics_df[(metrics_df["target"] == target) & (metrics_df["dataset"] == "test")].copy()
    order = ["baseline", "optimized"]
    target_df["model_type"] = pd.Categorical(target_df["model_type"], categories=order, ordered=True)
    target_df = target_df.sort_values("model_type")

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    metric_specs = [("rmse", "#ef4444"), ("mae", "#f59e0b"), ("r2", "#10b981")]
    for ax, (metric, color) in zip(axes, metric_specs):
        ax.bar(target_df["model_type"], target_df[metric], color=color)
        ax.set_title(f"{target} {metric.upper()}")
        ax.set_xlabel("Model Type")
        for idx, value in enumerate(target_df[metric]):
            ax.text(idx, value, f"{value:.4f}", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"{target}_metric_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()


def run_model(
    data: pd.DataFrame,
    target: str,
    model_type: str,
    features: list[str],
) -> tuple[pd.DataFrame, list[dict[str, object]], dict[str, object]]:
    X = data[features]
    y = data[target]
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
    )

    search = build_search()
    cv_splits = search.cv.n_splits
    search.fit(X_train, y_train)
    model: XGBRegressor = search.best_estimator_

    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    pred_df = pd.concat(
        [
            pd.DataFrame(
                {
                    "dataset": "train",
                    "actual": y_train.reset_index(drop=True),
                    "predicted": train_pred,
                    "residual": y_train.reset_index(drop=True) - train_pred,
                }
            ),
            pd.DataFrame(
                {
                    "dataset": "test",
                    "actual": y_test.reset_index(drop=True),
                    "predicted": test_pred,
                    "residual": y_test.reset_index(drop=True) - test_pred,
                }
            ),
        ],
        ignore_index=True,
    )
    pred_df.to_csv(
        OUTPUT_DIR / f"{target}_{model_type}_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )

    model.get_booster().save_model(OUTPUT_DIR / f"{target}_{model_type}_model.json")
    plot_actual_vs_pred(pred_df, target, model_type)
    plot_feature_importance(model, features, target, model_type)

    train_metrics = compute_metrics(y_train, train_pred)
    test_metrics = compute_metrics(y_test, test_pred)
    metric_rows = [
        {
            "target": target,
            "model_type": model_type,
            "dataset": "train",
            "feature_count": len(features),
            "features": ", ".join(features),
            "best_cv_rmse": float(-search.best_score_),
            **train_metrics,
        },
        {
            "target": target,
            "model_type": model_type,
            "dataset": "test",
            "feature_count": len(features),
            "features": ", ".join(features),
            "best_cv_rmse": float(-search.best_score_),
            **test_metrics,
        },
    ]

    model_info = {
        "target": target,
        "model_type": model_type,
        "features": features,
        "best_params": search.best_params_,
        "best_cv_rmse": float(-search.best_score_),
        "cv_splits": int(cv_splits),
        "train_size": int(len(X_train)),
        "test_size": int(len(X_test)),
    }
    return pred_df, metric_rows, model_info


def build_summary(metrics_df: pd.DataFrame, model_infos: list[dict[str, object]]) -> str:
    def get_test_row(target: str, model_type: str) -> pd.Series:
        return metrics_df[
            (metrics_df["target"] == target)
            & (metrics_df["model_type"] == model_type)
            & (metrics_df["dataset"] == "test")
        ].iloc[0]

    lines = [
        "# 第一问 XGBoost 建模结果摘要",
        "",
        "- 建模策略：采用“16个特征的基线模型 + GRA前5特征的优化模型”双模型框架。",
        "- 数据划分：按照 80% 训练集、20% 测试集进行随机划分。",
        "- 参数搜索：采用 RandomizedSearchCV，并使用 5 折 KFold 交叉验证。",
        "",
        "## 测试集表现",
    ]
    for target in TARGETS:
        base_row = get_test_row(target, "baseline")
        opt_row = get_test_row(target, "optimized")
        lines.append(
            f"- {target} 基线模型：RMSE={base_row['rmse']:.4f}，MAE={base_row['mae']:.4f}，R²={base_row['r2']:.4f}"
        )
        lines.append(
            f"- {target} 优化模型：RMSE={opt_row['rmse']:.4f}，MAE={opt_row['mae']:.4f}，R²={opt_row['r2']:.4f}"
        )
        winner = "优化模型" if opt_row["rmse"] <= base_row["rmse"] else "基线模型"
        lines.append(f"- 按 RMSE 指标判断，{target} 的较优模型为：{winner}。")
        lines.append("")

    lines.append("## 最优参数")
    for info in model_infos:
        lines.append(
            f"- {info['target']} {('基线模型' if info['model_type'] == 'baseline' else '优化模型')}："
            f"CV_RMSE={info['best_cv_rmse']:.4f}，"
            f"交叉验证折数={info['cv_splits']}，"
            f"最优参数={info['best_params']}"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    data_file = find_preprocessed_file()
    data = pd.read_csv(data_file, encoding="utf-8-sig")
    gra_feature_sets = load_gra_feature_sets()

    feature_manifest = pd.DataFrame(
        [
            {"target": "DFlow", "model_type": "baseline", "features": ", ".join(ALL_FEATURES)},
            {"target": "DFlow", "model_type": "optimized", "features": ", ".join(gra_feature_sets["DFlow"])},
            {"target": "DGap", "model_type": "baseline", "features": ", ".join(ALL_FEATURES)},
            {"target": "DGap", "model_type": "optimized", "features": ", ".join(gra_feature_sets["DGap"])},
        ]
    )
    feature_manifest.to_csv(OUTPUT_DIR / "feature_sets.csv", index=False, encoding="utf-8-sig")

    all_metric_rows: list[dict[str, object]] = []
    model_infos: list[dict[str, object]] = []

    for target in TARGETS:
        _, metric_rows, model_info = run_model(data, target, "baseline", ALL_FEATURES)
        all_metric_rows.extend(metric_rows)
        model_infos.append(model_info)

        _, metric_rows, model_info = run_model(data, target, "optimized", gra_feature_sets[target])
        all_metric_rows.extend(metric_rows)
        model_infos.append(model_info)

    metrics_df = pd.DataFrame(all_metric_rows)
    metrics_df.to_csv(OUTPUT_DIR / "model_metrics_summary.csv", index=False, encoding="utf-8-sig")

    with open(OUTPUT_DIR / "best_params.json", "w", encoding="utf-8") as f:
        json.dump(model_infos, f, ensure_ascii=False, indent=2)

    for target in TARGETS:
        plot_metric_comparison(metrics_df, target)

    summary = build_summary(metrics_df, model_infos)
    with open(OUTPUT_DIR / "Q1_XGBoost_summary.md", "w", encoding="utf-8-sig") as f:
        f.write(summary)

    print("XGBoost modeling completed.")


if __name__ == "__main__":
    main()
