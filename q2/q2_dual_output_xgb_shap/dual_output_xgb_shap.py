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
import shap
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.multioutput import MultiOutputRegressor
from xgboost import XGBRegressor


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
OUTPUT_DIR = SCRIPT_DIR / "dual_output_xgb_shap_outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH = OUTPUT_DIR / "dual_output_xgb_model.joblib"

plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial"]
plt.rcParams["axes.unicode_minus"] = False
sns.set_theme(style="whitegrid")

RANDOM_STATE = 42
FEATURES = [
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


def find_data_file() -> Path:
    candidates = [
        ROOT_DIR / "预处理后数据_未标准化.csv",
        ROOT_DIR / "数据预处理" / "output" / "预处理后数据_未标准化.csv",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError("未找到预处理后数据_未标准化.csv")


def build_model(params: dict[str, float | int]) -> MultiOutputRegressor:
    base = XGBRegressor(
        objective="reg:squarederror",
        random_state=RANDOM_STATE,
        eval_metric="rmse",
        n_jobs=1,
        **params,
    )
    return MultiOutputRegressor(base, n_jobs=1)


def combined_metrics(y_true: pd.DataFrame, y_pred: np.ndarray) -> tuple[dict[str, float], list[dict[str, float]]]:
    overall = {
        "joint_rmse_mean": float(np.mean([np.sqrt(mean_squared_error(y_true.iloc[:, i], y_pred[:, i])) for i in range(2)])),
        "joint_mae_mean": float(np.mean([mean_absolute_error(y_true.iloc[:, i], y_pred[:, i]) for i in range(2)])),
        "joint_r2_mean": float(np.mean([r2_score(y_true.iloc[:, i], y_pred[:, i]) for i in range(2)])),
    }
    by_target = []
    for i, target in enumerate(TARGETS):
        by_target.append(
            {
                "target": target,
                "rmse": float(np.sqrt(mean_squared_error(y_true[target], y_pred[:, i]))),
                "mae": float(mean_absolute_error(y_true[target], y_pred[:, i])),
                "r2": float(r2_score(y_true[target], y_pred[:, i])),
            }
        )
    return overall, by_target


def search_best_params(X: pd.DataFrame, y: pd.DataFrame) -> tuple[dict[str, float | int], pd.DataFrame]:
    param_grid = [
        {"n_estimators": 60, "max_depth": 2, "learning_rate": 0.05, "subsample": 0.8, "colsample_bytree": 0.8, "reg_lambda": 1.0, "gamma": 0.0},
        {"n_estimators": 80, "max_depth": 2, "learning_rate": 0.05, "subsample": 0.9, "colsample_bytree": 0.8, "reg_lambda": 1.0, "gamma": 0.1},
        {"n_estimators": 100, "max_depth": 3, "learning_rate": 0.05, "subsample": 0.8, "colsample_bytree": 0.8, "reg_lambda": 3.0, "gamma": 0.1},
        {"n_estimators": 120, "max_depth": 3, "learning_rate": 0.03, "subsample": 0.9, "colsample_bytree": 0.9, "reg_lambda": 3.0, "gamma": 0.0},
        {"n_estimators": 150, "max_depth": 4, "learning_rate": 0.03, "subsample": 0.8, "colsample_bytree": 0.8, "reg_lambda": 5.0, "gamma": 0.1},
        {"n_estimators": 100, "max_depth": 4, "learning_rate": 0.08, "subsample": 0.9, "colsample_bytree": 0.9, "reg_lambda": 1.0, "gamma": 0.2},
    ]
    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    rows = []
    for idx, params in enumerate(param_grid, start=1):
        fold_scores = []
        for fold_id, (tr_idx, va_idx) in enumerate(cv.split(X), start=1):
            model = build_model(params)
            model.fit(X.iloc[tr_idx], y.iloc[tr_idx])
            pred = model.predict(X.iloc[va_idx])
            overall, by_target = combined_metrics(y.iloc[va_idx], pred)
            row = {"param_id": idx, "fold_id": fold_id, **params, **overall}
            for item in by_target:
                row[f"{item['target']}_rmse"] = item["rmse"]
                row[f"{item['target']}_mae"] = item["mae"]
                row[f"{item['target']}_r2"] = item["r2"]
            fold_scores.append(row)
        rows.extend(fold_scores)
    cv_df = pd.DataFrame(rows)
    cv_df.to_csv(OUTPUT_DIR / "dual_output_cv_folds.csv", index=False, encoding="utf-8-sig")
    summary = cv_df.groupby("param_id").agg(
        joint_rmse_mean=("joint_rmse_mean", "mean"),
        joint_rmse_std=("joint_rmse_mean", "std"),
        joint_r2_mean=("joint_r2_mean", "mean"),
        DFlow_rmse_mean=("DFlow_rmse", "mean"),
        DGap_rmse_mean=("DGap_rmse", "mean"),
    ).reset_index()
    first_rows = cv_df.groupby("param_id").first().reset_index()
    cv_summary = summary.merge(
        first_rows[["param_id", "n_estimators", "max_depth", "learning_rate", "subsample", "colsample_bytree", "reg_lambda", "gamma"]],
        on="param_id",
        how="left",
    ).sort_values("joint_rmse_mean")
    cv_summary.to_csv(OUTPUT_DIR / "dual_output_cv_summary.csv", index=False, encoding="utf-8-sig")
    best_param_id = int(cv_summary.iloc[0]["param_id"])
    best_params = param_grid[best_param_id - 1]
    return best_params, cv_summary


def plot_actual_vs_pred(y_true: pd.DataFrame, y_pred: np.ndarray) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for i, target in enumerate(TARGETS):
        ax = axes[i]
        sns.scatterplot(x=y_true[target], y=y_pred[:, i], ax=ax, s=55)
        lims = [min(y_true[target].min(), y_pred[:, i].min()), max(y_true[target].max(), y_pred[:, i].max())]
        ax.plot(lims, lims, "--", color="black", linewidth=1)
        ax.set_title(f"{target}: Actual vs Predicted")
        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "dual_output_actual_vs_pred.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_metrics(metrics_df: pd.DataFrame) -> None:
    plt.figure(figsize=(8, 5))
    melted = metrics_df.melt(id_vars=["target"], value_vars=["rmse", "mae", "r2"], var_name="metric", value_name="value")
    sns.barplot(data=melted, x="target", y="value", hue="metric")
    plt.title("Dual-output Test Metrics")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "dual_output_metrics.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_feature_importance(importance_df: pd.DataFrame) -> None:
    plt.figure(figsize=(12, 6))
    sns.barplot(data=importance_df, x="feature", y="importance", hue="output")
    plt.xticks(rotation=35, ha="right")
    plt.title("Dual-output XGBoost Feature Importance")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "dual_output_feature_importance.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_synergy_indices(synergy_df: pd.DataFrame) -> None:
    plot_df = synergy_df.sort_values("shared_contribution", ascending=False)
    plt.figure(figsize=(12, 6))
    x = np.arange(len(plot_df))
    width = 0.35
    plt.bar(x - width / 2, plot_df["shared_contribution"], width=width, label="SharedContribution")
    plt.bar(x + width / 2, plot_df["differential_contribution"], width=width, label="DifferentialContribution")
    plt.xticks(x, plot_df["feature"], rotation=35, ha="right")
    plt.legend()
    plt.title("Shared vs Differential Contribution")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "shared_vs_differential_drivers.png", dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(12, 5))
    direction_df = synergy_df.sort_values("direction_correlation")
    colors = ["#ef4444" if v < 0 else "#10b981" for v in direction_df["direction_correlation"]]
    plt.bar(direction_df["feature"], direction_df["direction_correlation"], color=colors)
    plt.xticks(rotation=35, ha="right")
    plt.axhline(0, color="black", linewidth=1)
    plt.title("Direction Synergy Correlation")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "direction_synergy_correlation.png", dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=synergy_df, x="shared_contribution", y="differential_contribution", size="abs_direction_correlation", hue="direction_correlation", palette="coolwarm", sizes=(50, 250))
    for _, row in synergy_df.iterrows():
        plt.text(row["shared_contribution"] + 0.002, row["differential_contribution"] + 0.002, row["feature"], fontsize=8)
    plt.title("Shared-Differential Driver Map")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "shared_differential_driver_map.png", dpi=300, bbox_inches="tight")
    plt.close()


def save_shap_plots(estimator: XGBRegressor, X: pd.DataFrame, output_name: str) -> tuple[np.ndarray, float]:
    explainer = shap.TreeExplainer(estimator)
    shap_values = np.array(explainer.shap_values(X))
    expected_value = explainer.expected_value
    if isinstance(expected_value, np.ndarray):
        expected_value = float(np.ravel(expected_value)[0])
    plt.figure()
    shap.summary_plot(shap_values, X, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"{output_name}_shap_bar.png", dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure()
    shap.summary_plot(shap_values, X, show=False)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"{output_name}_shap_beeswarm.png", dpi=300, bbox_inches="tight")
    plt.close()
    return shap_values, expected_value


def save_dependence_plot(shap_values: np.ndarray, X: pd.DataFrame, feature: str, output_name: str) -> None:
    plt.figure()
    shap.dependence_plot(feature, shap_values, X, interaction_index=None, show=False)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"{output_name}_dependence_{feature}.png", dpi=300, bbox_inches="tight")
    plt.close()


def save_waterfall_plot(shap_values: np.ndarray, expected_value: float, X: pd.DataFrame, sample_index: int, output_name: str, suffix: str) -> None:
    explanation = shap.Explanation(
        values=shap_values[sample_index],
        base_values=expected_value,
        data=X.iloc[sample_index].values,
        feature_names=list(X.columns),
    )
    plt.figure()
    shap.plots.waterfall(explanation, max_display=10, show=False)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"{output_name}_waterfall_{suffix}.png", dpi=300, bbox_inches="tight")
    plt.close()


def compute_shap_values(estimator: XGBRegressor, X: pd.DataFrame) -> tuple[np.ndarray, float]:
    explainer = shap.TreeExplainer(estimator)
    shap_values = np.array(explainer.shap_values(X))
    expected_value = explainer.expected_value
    if isinstance(expected_value, np.ndarray):
        expected_value = float(np.ravel(expected_value)[0])
    return shap_values, float(expected_value)


def main() -> None:
    data = pd.read_csv(find_data_file(), encoding="utf-8-sig")
    X = data[FEATURES]
    y = data[TARGETS]

    best_params, cv_summary = search_best_params(X, y)
    with open(OUTPUT_DIR / "best_params.json", "w", encoding="utf-8") as f:
        json.dump(best_params, f, ensure_ascii=False, indent=2)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)
    model = build_model(best_params)
    model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    overall_train, train_by_target = combined_metrics(y_train, y_train_pred)
    overall_test, test_by_target = combined_metrics(y_test, y_test_pred)
    metric_rows = []
    for split, overall, rows in [("train", overall_train, train_by_target), ("test", overall_test, test_by_target)]:
        for row in rows:
            metric_rows.append({"split": split, **row, **overall})
    metrics_df = pd.DataFrame(metric_rows)
    metrics_df.to_csv(OUTPUT_DIR / "dual_output_metric_summary.csv", index=False, encoding="utf-8-sig")

    pred_df = pd.DataFrame(
        {
            "DFlow_actual": y_test["DFlow"].values,
            "DFlow_predicted": y_test_pred[:, 0],
            "DGap_actual": y_test["DGap"].values,
            "DGap_predicted": y_test_pred[:, 1],
        }
    )
    pred_df["joint_prediction_sum"] = pred_df["DFlow_predicted"] + pred_df["DGap_predicted"]
    pred_df.to_csv(OUTPUT_DIR / "dual_output_test_predictions.csv", index=False, encoding="utf-8-sig")

    plot_actual_vs_pred(y_test, y_test_pred)
    plot_metrics(pd.DataFrame(test_by_target))

    importance_rows = []
    shap_rows = []
    local_rows = []
    for idx, target in enumerate(TARGETS):
        estimator = model.estimators_[idx]
        for feature, importance in zip(FEATURES, estimator.feature_importances_):
            importance_rows.append({"output": target, "feature": feature, "importance": float(importance)})

        shap_values_full, expected_value = save_shap_plots(estimator, X, target)
        shap_values_test, expected_value_test = compute_shap_values(estimator, X_test)
        mean_abs = np.abs(shap_values_full).mean(axis=0)
        mean_signed = shap_values_full.mean(axis=0)
        shap_df = pd.DataFrame(
            {
                "output": target,
                "feature": FEATURES,
                "mean_abs_shap": mean_abs,
                "mean_signed_shap": mean_signed,
                "shap_rank": pd.Series(mean_abs).rank(ascending=False, method="min").astype(int).values,
            }
        ).sort_values("shap_rank")
        shap_rows.append(shap_df)

        top_features = shap_df.head(2)["feature"].tolist()
        for feature in top_features:
            save_dependence_plot(shap_values_full, X, feature, target)

        if target == "DFlow":
            high_idx = int(np.argmax(y_test_pred[:, 0]))
            low_idx = int(np.argmin(y_test_pred[:, 0]))
        else:
            high_idx = int(np.argmax(y_test_pred[:, 1]))
            low_idx = int(np.argmin(y_test_pred[:, 1]))
        save_waterfall_plot(shap_values_test, expected_value_test, X_test.reset_index(drop=True), high_idx, target, "high_prediction")
        save_waterfall_plot(shap_values_test, expected_value_test, X_test.reset_index(drop=True), low_idx, target, "low_prediction")

        for tag, sample_idx in [("high_prediction", high_idx), ("low_prediction", low_idx)]:
            sample_df = pd.DataFrame(
                {
                    "output": target,
                    "case": tag,
                    "sample_index": sample_idx,
                    "feature": FEATURES,
                    "feature_value": X_test.reset_index(drop=True).iloc[sample_idx].values,
                    "shap_value": shap_values_test[sample_idx],
                    "expected_value": expected_value_test,
                }
            ).sort_values("shap_value", key=lambda s: s.abs(), ascending=False)
            local_rows.append(sample_df)

    importance_df = pd.DataFrame(importance_rows)
    importance_df.to_csv(OUTPUT_DIR / "dual_output_feature_importance.csv", index=False, encoding="utf-8-sig")
    plot_feature_importance(importance_df)

    shap_importance_df = pd.concat(shap_rows, ignore_index=True)
    shap_importance_df.to_csv(OUTPUT_DIR / "dual_output_shap_importance.csv", index=False, encoding="utf-8-sig")
    pd.concat(local_rows, ignore_index=True).to_csv(OUTPUT_DIR / "dual_output_local_explanations.csv", index=False, encoding="utf-8-sig")

    shap_flow, _ = compute_shap_values(model.estimators_[0], X)
    shap_gap, _ = compute_shap_values(model.estimators_[1], X)
    shared = (np.abs(shap_flow) + np.abs(shap_gap)) / 2
    differential = np.abs(np.abs(shap_flow) - np.abs(shap_gap))
    synergy_rows = []
    for j, feature in enumerate(FEATURES):
        corr = np.corrcoef(shap_flow[:, j], shap_gap[:, j])[0, 1]
        if np.isnan(corr):
            corr = 0.0
        synergy_rows.append(
            {
                "feature": feature,
                "shared_contribution": float(shared[:, j].mean()),
                "differential_contribution": float(differential[:, j].mean()),
                "direction_correlation": float(corr),
                "abs_direction_correlation": float(abs(corr)),
                "driver_type": "shared_driver" if shared[:, j].mean() >= np.quantile(shared.mean(axis=0), 0.75) and differential[:, j].mean() < np.quantile(differential.mean(axis=0), 0.5) else (
                    "compensatory_driver" if corr < -0.2 else (
                        "differential_driver" if differential[:, j].mean() >= np.quantile(differential.mean(axis=0), 0.75) else "mixed_driver"
                    )
                ),
            }
        )
    synergy_df = pd.DataFrame(synergy_rows).sort_values("shared_contribution", ascending=False)
    synergy_df.to_csv(OUTPUT_DIR / "dual_output_synergy_indices.csv", index=False, encoding="utf-8-sig")
    plot_synergy_indices(synergy_df)

    shared_top = synergy_df.sort_values("shared_contribution", ascending=False).head(5)["feature"].tolist()
    diff_top = synergy_df.sort_values("differential_contribution", ascending=False).head(5)["feature"].tolist()
    comp_top = synergy_df[synergy_df["direction_correlation"] < 0].sort_values("direction_correlation").head(5)["feature"].tolist()

    summary_lines = [
        "# 第二问 双输出 XGBoost + SHAP 协同解释模型结果摘要",
        "",
        "- 建模目标：以 16 个脉冲特征为输入，基于 MultiOutputRegressor 封装两个 XGBoost 回归器，对 (DFlow, DGap) 做双输出联合分析，并用 SHAP 分解共享驱动、差异驱动与补偿驱动。",
        f"- 最优参数：{best_params}",
        "",
        "## 测试集性能",
    ]
    for row in test_by_target:
        summary_lines.append(
            f"- {row['target']}：RMSE={row['rmse']:.4f}，MAE={row['mae']:.4f}，R²={row['r2']:.4f}"
        )
    summary_lines.extend(
        [
            f"- 综合平均 RMSE={overall_test['joint_rmse_mean']:.4f}，综合平均 R²={overall_test['joint_r2_mean']:.4f}",
            "",
            "## 协同解释结果",
            f"- 共享驱动因子前 5 名：{', '.join(shared_top)}",
            f"- 差异驱动因子前 5 名：{', '.join(diff_top)}",
            f"- 补偿型特征（方向协同系数为负且绝对值较大）前 5 名：{', '.join(comp_top) if comp_top else '无明显补偿因子'}",
            "",
            "## 解释结论",
            "- 该模型在实现上属于“共享输入空间 + 双输出并行回归器”的联合分析框架，而非原生单损失的多目标树模型。",
            "- 若某特征 shared_contribution 高且 differential_contribution 低，则说明它同时影响 DFlow 与 DGap，是协同调控的共享监测指标。",
            "- 若某特征 differential_contribution 高，则说明它更偏向于其中一个控制变量，是差异驱动因子。",
            "- 若某特征 direction_correlation 为负，则说明它对两个输出的 SHAP 方向存在补偿关系，可用于解释主从协同回归中出现的负向联动项。",
        ]
    )
    with open(OUTPUT_DIR / "Q2_dual_output_xgb_shap_summary.md", "w", encoding="utf-8-sig") as f:
        f.write("\n".join(summary_lines) + "\n")

    final_model = build_model(best_params)
    final_model.fit(X, y)
    joblib.dump(final_model, MODEL_PATH)

    print("Q2 dual-output XGBoost + SHAP completed.")


if __name__ == "__main__":
    main()
