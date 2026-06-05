from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from xgboost import XGBRegressor


SCRIPT_DIR = Path(__file__).resolve().parent
Q1_DIR = SCRIPT_DIR.parent
ROOT_DIR = Q1_DIR.parent
XGB_OUTPUT_DIR = Q1_DIR / "xgboost_q1" / "outputs"
GRA_OUTPUT_DIR = Q1_DIR / "gra_q1" / "outputs"
OUTPUT_DIR = SCRIPT_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial"]
plt.rcParams["axes.unicode_minus"] = False
sns.set_theme(style="whitegrid")

PRIMARY_MODELS = [
    {"target": "DFlow", "model_type": "optimized"},
    {"target": "DGap", "model_type": "baseline"},
]


def find_preprocessed_file() -> Path:
    candidates = [
        ROOT_DIR / "预处理后数据_未标准化.csv",
        ROOT_DIR / "数据预处理" / "output" / "预处理后数据_未标准化.csv",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError("未找到预处理后数据_未标准化.csv")


def load_feature_sets() -> dict[tuple[str, str], list[str]]:
    feature_sets = pd.read_csv(XGB_OUTPUT_DIR / "feature_sets.csv", encoding="utf-8-sig")
    mapping: dict[tuple[str, str], list[str]] = {}
    for _, row in feature_sets.iterrows():
        mapping[(row["target"], row["model_type"])] = [item.strip() for item in row["features"].split(",")]
    return mapping


def load_gra_results(target: str) -> pd.DataFrame:
    return pd.read_csv(GRA_OUTPUT_DIR / f"{target}_gra_results.csv", encoding="utf-8-sig")


def load_xgb_importance(target: str, model_type: str) -> pd.DataFrame:
    return pd.read_csv(
        XGB_OUTPUT_DIR / f"{target}_{model_type}_feature_importance.csv",
        encoding="utf-8-sig",
    )


def load_model(target: str, model_type: str) -> XGBRegressor:
    model = XGBRegressor()
    model.load_model(XGB_OUTPUT_DIR / f"{target}_{model_type}_model.json")
    return model


def compute_shap_values(
    model: XGBRegressor,
    X: pd.DataFrame,
) -> tuple[np.ndarray, float]:
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    expected_value = explainer.expected_value
    if isinstance(expected_value, np.ndarray):
        expected_value = float(np.ravel(expected_value)[0])
    return np.array(shap_values), float(expected_value)


def save_summary_bar(shap_values: np.ndarray, X: pd.DataFrame, prefix: str) -> None:
    plt.figure()
    shap.summary_plot(shap_values, X, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"{prefix}_shap_bar.png", dpi=300, bbox_inches="tight")
    plt.close()


def save_beeswarm(shap_values: np.ndarray, X: pd.DataFrame, prefix: str) -> None:
    plt.figure()
    shap.summary_plot(shap_values, X, show=False)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"{prefix}_shap_beeswarm.png", dpi=300, bbox_inches="tight")
    plt.close()


def save_dependence(shap_values: np.ndarray, X: pd.DataFrame, feature: str, prefix: str) -> None:
    plt.figure()
    shap.dependence_plot(feature, shap_values, X, show=False, interaction_index=None)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"{prefix}_dependence_{feature}.png", dpi=300, bbox_inches="tight")
    plt.close()


def save_waterfall(
    shap_values: np.ndarray,
    X: pd.DataFrame,
    expected_value: float,
    sample_index: int,
    prefix: str,
    tag: str,
) -> None:
    explanation = shap.Explanation(
        values=shap_values[sample_index],
        base_values=expected_value,
        data=X.iloc[sample_index].values,
        feature_names=list(X.columns),
    )
    plt.figure()
    shap.plots.waterfall(explanation, max_display=min(10, X.shape[1]), show=False)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"{prefix}_waterfall_{tag}.png", dpi=300, bbox_inches="tight")
    plt.close()


def build_importance_table(
    target: str,
    model_type: str,
    shap_values: np.ndarray,
    X: pd.DataFrame,
) -> pd.DataFrame:
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    mean_signed_shap = shap_values.mean(axis=0)
    result = pd.DataFrame(
        {
            "feature": X.columns,
            "mean_abs_shap": mean_abs_shap,
            "mean_signed_shap": mean_signed_shap,
            "shap_rank": pd.Series(mean_abs_shap).rank(ascending=False, method="min").astype(int).values,
        }
    ).sort_values("shap_rank")
    result["target"] = target
    result["model_type"] = model_type
    result["direction"] = result["mean_signed_shap"].apply(lambda x: "Positive" if x >= 0 else "Negative")
    return result


def build_comparison_table(
    target: str,
    model_type: str,
    shap_importance: pd.DataFrame,
) -> pd.DataFrame:
    gra_df = load_gra_results(target)[["feature", "rank", "gra_degree"]].rename(
        columns={"rank": "gra_rank", "gra_degree": "gra_degree"}
    )
    xgb_df = load_xgb_importance(target, model_type)
    xgb_df["xgb_rank"] = xgb_df["importance"].rank(ascending=False, method="min").astype(int)
    comparison = shap_importance.merge(gra_df, on="feature", how="left").merge(
        xgb_df[["feature", "importance", "xgb_rank"]],
        on="feature",
        how="left",
    )
    comparison = comparison.sort_values("shap_rank").reset_index(drop=True)
    comparison["target"] = target
    comparison["model_type"] = model_type
    return comparison[
        [
            "target",
            "model_type",
            "feature",
            "gra_rank",
            "gra_degree",
            "xgb_rank",
            "importance",
            "shap_rank",
            "mean_abs_shap",
            "mean_signed_shap",
            "direction",
        ]
    ]


def build_local_contribution_table(
    target: str,
    model_type: str,
    shap_values: np.ndarray,
    expected_value: float,
    X: pd.DataFrame,
    y: pd.Series,
    predictions: np.ndarray,
) -> pd.DataFrame:
    selected_indices = {
        "high_prediction": int(np.argmax(predictions)),
        "low_prediction": int(np.argmin(predictions)),
    }
    rows = []
    for tag, idx in selected_indices.items():
        feature_df = pd.DataFrame(
            {
                "feature": X.columns,
                "feature_value": X.iloc[idx].values,
                "shap_value": shap_values[idx],
            }
        ).sort_values("shap_value", key=lambda s: s.abs(), ascending=False)
        feature_df.insert(0, "case", tag)
        feature_df.insert(1, "sample_index", idx)
        feature_df.insert(2, "target", target)
        feature_df.insert(3, "model_type", model_type)
        feature_df.insert(4, "base_value", expected_value)
        feature_df.insert(5, "prediction", float(predictions[idx]))
        feature_df.insert(6, "actual", float(y.iloc[idx]))
        rows.append(feature_df)
    return pd.concat(rows, ignore_index=True)


def build_summary(model_summaries: list[dict[str, object]]) -> str:
    lines = [
        "# 第一问 SHAP 解释结果摘要",
        "",
        "- 方法定位：在完成 GRA 与 XGBoost 回归建模后，使用 SHAP 对最优或较优模型进行后解释。",
        "- 重点解释对象：DFlow 优化模型、DGap 基线模型。",
        "- SHAP 含义：将单个样本预测值分解为“基线值 + 各特征贡献值”之和，从而解释模型为什么这样预测。",
        "",
    ]
    for item in model_summaries:
        lines.append(f"## {item['target']} {item['model_name']} 的主要解释结论")
        lines.append(f"- 基线值（expected value）：{item['expected_value']:.4f}")
        lines.append(f"- 平均绝对 SHAP 值排名前 5 的特征：{', '.join(item['top5_features'])}")
        lines.append(
            f"- 总体正向推动较明显的特征：{', '.join(item['positive_features']) if item['positive_features'] else '无'}"
        )
        lines.append(
            f"- 总体负向推动较明显的特征：{', '.join(item['negative_features']) if item['negative_features'] else '无'}"
        )
        lines.append(
            f"- 高预测样本编号：{item['high_sample']}；低预测样本编号：{item['low_sample']}。"
        )
        lines.append("")
    lines.append("## 解释层面的总体结论")
    lines.append("- DFlow 更适合使用 GRA 筛选后的少量关键特征进行解释，说明其主导驱动因子较集中。")
    lines.append("- DGap 更依赖全特征信息，说明前五特征之外仍有变量在模型中提供补充解释力。")
    lines.append("- GRA 排序反映趋势接近性，XGBoost 重要性反映树分裂收益，而 SHAP 更直接反映模型输出贡献。")
    return "\n".join(lines) + "\n"


def main() -> None:
    raw_file = find_preprocessed_file()
    raw_df = pd.read_csv(raw_file, encoding="utf-8-sig")
    feature_sets = load_feature_sets()

    manifest_rows = []
    all_shap_importance = []
    all_comparisons = []
    all_local_rows = []
    summary_rows = []

    for config in PRIMARY_MODELS:
        target = config["target"]
        model_type = config["model_type"]
        model_name = "优化模型" if model_type == "optimized" else "基线模型"
        features = feature_sets[(target, model_type)]
        model = load_model(target, model_type)
        X = raw_df[features]
        y = raw_df[target]
        predictions = model.predict(X)
        shap_values, expected_value = compute_shap_values(model, X)

        prefix = f"{target}_{model_type}"
        save_summary_bar(shap_values, X, prefix)
        save_beeswarm(shap_values, X, prefix)

        shap_importance = build_importance_table(target, model_type, shap_values, X)
        shap_importance.to_csv(
            OUTPUT_DIR / f"{prefix}_shap_importance.csv",
            index=False,
            encoding="utf-8-sig",
        )
        all_shap_importance.append(shap_importance)

        top_features = shap_importance.sort_values("shap_rank")["feature"].head(2).tolist()
        for feature in top_features:
            save_dependence(shap_values, X, feature, prefix)

        high_idx = int(np.argmax(predictions))
        low_idx = int(np.argmin(predictions))
        save_waterfall(shap_values, X, expected_value, high_idx, prefix, "high_prediction")
        save_waterfall(shap_values, X, expected_value, low_idx, prefix, "low_prediction")

        local_rows = build_local_contribution_table(
            target, model_type, shap_values, expected_value, X, y, predictions
        )
        local_rows.to_csv(
            OUTPUT_DIR / f"{prefix}_local_explanations.csv",
            index=False,
            encoding="utf-8-sig",
        )
        all_local_rows.append(local_rows)

        comparison = build_comparison_table(target, model_type, shap_importance)
        comparison.to_csv(
            OUTPUT_DIR / f"{prefix}_gra_xgb_shap_comparison.csv",
            index=False,
            encoding="utf-8-sig",
        )
        all_comparisons.append(comparison)

        manifest_rows.append(
            {
                "target": target,
                "model_type": model_type,
                "model_name": model_name,
                "features": ", ".join(features),
                "expected_value": expected_value,
                "high_prediction_sample_index": high_idx,
                "low_prediction_sample_index": low_idx,
            }
        )
        summary_rows.append(
            {
                "target": target,
                "model_name": model_name,
                "expected_value": expected_value,
                "top5_features": shap_importance.sort_values("shap_rank")["feature"].head(5).tolist(),
                "positive_features": shap_importance[shap_importance["mean_signed_shap"] > 0]
                .sort_values("mean_abs_shap", ascending=False)["feature"]
                .head(3)
                .tolist(),
                "negative_features": shap_importance[shap_importance["mean_signed_shap"] < 0]
                .sort_values("mean_abs_shap", ascending=False)["feature"]
                .head(3)
                .tolist(),
                "high_sample": high_idx,
                "low_sample": low_idx,
            }
        )

    pd.DataFrame(manifest_rows).to_csv(
        OUTPUT_DIR / "shap_analysis_manifest.csv", index=False, encoding="utf-8-sig"
    )
    pd.concat(all_shap_importance, ignore_index=True).to_csv(
        OUTPUT_DIR / "shap_importance_combined.csv", index=False, encoding="utf-8-sig"
    )
    pd.concat(all_comparisons, ignore_index=True).to_csv(
        OUTPUT_DIR / "gra_xgb_shap_comparison_combined.csv", index=False, encoding="utf-8-sig"
    )
    pd.concat(all_local_rows, ignore_index=True).to_csv(
        OUTPUT_DIR / "local_explanations_combined.csv", index=False, encoding="utf-8-sig"
    )

    summary = build_summary(summary_rows)
    with open(OUTPUT_DIR / "Q1_SHAP_summary.md", "w", encoding="utf-8-sig") as f:
        f.write(summary)

    with open(OUTPUT_DIR / "shap_summary_meta.json", "w", encoding="utf-8") as f:
        json.dump(summary_rows, f, ensure_ascii=False, indent=2)

    print("SHAP analysis completed.")


if __name__ == "__main__":
    main()
