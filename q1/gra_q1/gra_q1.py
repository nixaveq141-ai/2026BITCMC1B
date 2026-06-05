from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


BASE_DIR = Path(__file__).resolve().parent
PREPROCESS_DIR = BASE_DIR.parent / "数据预处理" / "output"
OUTPUT_DIR = BASE_DIR
DISTINGUISHING_COEFFICIENT = 0.5

plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial"]
plt.rcParams["axes.unicode_minus"] = False
sns.set_theme(style="whitegrid")

FEATURE_COLUMNS = [
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
TARGET_COLUMNS = ["DFlow", "DGap"]


def min_max_scale(series: pd.Series) -> pd.Series:
    span = series.max() - series.min()
    if span == 0:
        return pd.Series(0.0, index=series.index, name=series.name)
    return (series - series.min()) / span


def gra_for_target(minmax_df: pd.DataFrame, raw_df: pd.DataFrame, target: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    reference = minmax_df[target]
    compare = minmax_df[FEATURE_COLUMNS]
    delta = compare.sub(reference, axis=0).abs()
    delta_min = delta.min().min()
    delta_max = delta.max().max()
    coeff = (delta_min + DISTINGUISHING_COEFFICIENT * delta_max) / (
        delta + DISTINGUISHING_COEFFICIENT * delta_max
    )
    gra_degree = coeff.mean(axis=0)

    spearman = raw_df[FEATURE_COLUMNS + [target]].corr(method="spearman")[target].drop(target)
    result = pd.DataFrame(
        {
            "feature": FEATURE_COLUMNS,
            "gra_degree": [gra_degree[col] for col in FEATURE_COLUMNS],
            "spearman_rho": [spearman[col] for col in FEATURE_COLUMNS],
        }
    )
    result["sign"] = result["spearman_rho"].apply(lambda x: 1 if x >= 0 else -1)
    result["signed_influence"] = result["gra_degree"] * result["sign"]
    result["rank"] = result["gra_degree"].rank(ascending=False, method="min").astype(int)
    result["relationship_direction"] = result["spearman_rho"].apply(lambda x: "Positive" if x >= 0 else "Negative")
    result["association_level"] = pd.cut(
        result["gra_degree"],
        bins=[-1, 0.50, 0.65, 0.80, 1.01],
        labels=["Weak", "Medium", "Strong", "Very Strong"],
    )
    result = result.sort_values(["rank", "feature"]).reset_index(drop=True)

    coeff_export = coeff.copy()
    coeff_export.insert(0, "sample_index", range(len(coeff_export)))
    return result, coeff_export


def plot_gra_bar(result: pd.DataFrame, target: str) -> None:
    plot_df = result.sort_values("gra_degree", ascending=True)
    plt.figure(figsize=(10, 7))
    colors = ["#3b82f6" if v >= 0 else "#ef4444" for v in plot_df["signed_influence"]]
    plt.barh(plot_df["feature"], plot_df["gra_degree"], color=colors)
    plt.xlabel("Grey Relational Degree")
    plt.ylabel("Feature")
    plt.title(f"GRA Ranking for {target}")
    for idx, value in enumerate(plot_df["gra_degree"]):
        plt.text(value + 0.002, idx, f"{value:.4f}", va="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"{target}_gra_ranking.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_signed_influence(result: pd.DataFrame, target: str) -> None:
    plot_df = result.sort_values("signed_influence")
    plt.figure(figsize=(10, 7))
    colors = ["#ef4444" if v < 0 else "#10b981" for v in plot_df["signed_influence"]]
    plt.barh(plot_df["feature"], plot_df["signed_influence"], color=colors)
    plt.axvline(0, color="black", linewidth=1)
    plt.xlabel("Signed Influence Index")
    plt.ylabel("Feature")
    plt.title(f"Signed Influence Index for {target}")
    for idx, value in enumerate(plot_df["signed_influence"]):
        offset = -0.01 if value < 0 else 0.002
        plt.text(value + offset, idx, f"{value:.4f}", va="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"{target}_signed_influence.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_coefficient_heatmap(coeff_df: pd.DataFrame, target: str) -> None:
    coeff_only = coeff_df[FEATURE_COLUMNS]
    summary_df = pd.DataFrame(
        {
            "mean": coeff_only.mean(axis=0),
            "median": coeff_only.median(axis=0),
            "std": coeff_only.std(axis=0),
            "min": coeff_only.min(axis=0),
            "max": coeff_only.max(axis=0),
        }
    ).loc[FEATURE_COLUMNS]

    plt.figure(figsize=(8, 9))
    sns.heatmap(summary_df, cmap="YlGnBu", annot=True, fmt=".3f", cbar=True)
    plt.xlabel("Aggregated Statistics")
    plt.ylabel("Feature")
    plt.title(f"Aggregated Grey Relational Coefficients for {target}")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"{target}_coefficient_heatmap.png", dpi=300, bbox_inches="tight")
    plt.close()

    summary_df.reset_index().rename(columns={"index": "feature"}).to_csv(
        OUTPUT_DIR / f"{target}_coefficient_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )


def plot_target_comparison(dflow_result: pd.DataFrame, dgap_result: pd.DataFrame) -> None:
    merged = dflow_result[["feature", "gra_degree"]].merge(
        dgap_result[["feature", "gra_degree"]],
        on="feature",
        suffixes=("_DFlow", "_DGap"),
    )
    plot_df = merged.sort_values("gra_degree_DFlow", ascending=True)
    fig, axes = plt.subplots(1, 2, figsize=(16, 8), sharey=True)
    axes[0].barh(plot_df["feature"], plot_df["gra_degree_DFlow"], color="#3b82f6")
    axes[0].set_title("DFlow Grey Relational Degree")
    axes[0].set_xlabel("GRA Degree")
    axes[1].barh(plot_df["feature"], plot_df["gra_degree_DGap"], color="#10b981")
    axes[1].set_title("DGap Grey Relational Degree")
    axes[1].set_xlabel("GRA Degree")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "DFlow_DGap_gra_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()


def build_summary(dflow_result: pd.DataFrame, dgap_result: pd.DataFrame, source_file: Path) -> str:
    def top_lines(df: pd.DataFrame, target: str) -> list[str]:
        top5 = df.nsmallest(5, "rank")
        lines = [f"## {target} 前五特征"]
        for _, row in top5.iterrows():
            lines.append(
                f"- {row['feature']}：灰色关联度={row['gra_degree']:.4f}，"
                f"Spearman={row['spearman_rho']:.4f}，"
                f"带符号影响指数={row['signed_influence']:.4f}，"
                f"方向={row['relationship_direction']}"
            )
        return lines

    common_top = set(dflow_result.nsmallest(5, "rank")["feature"]).intersection(
        set(dgap_result.nsmallest(5, "rank")["feature"])
    )
    summary_lines = [
        "# 问题一灰色关联分析结果",
        "",
        f"- 数据来源：{source_file.name}",
        "- 方法：极差标准化 + 灰色关联分析（GRA） + Spearman方向增强",
        f"- 分辨系数：{DISTINGUISHING_COEFFICIENT}",
        "- 输入特征：16个脉冲特征参数",
        "- 目标变量：DFlow、DGap",
        "",
    ]
    summary_lines.extend(top_lines(dflow_result, "DFlow"))
    summary_lines.append("")
    summary_lines.extend(top_lines(dgap_result, "DGap"))
    summary_lines.append("")
    summary_lines.append("## 结果解读")
    summary_lines.append(
        f"- 两个目标变量前五集合的共同关键特征为：{', '.join(sorted(common_top)) if common_top else '无'}。"
    )
    summary_lines.append(
        "- 带符号影响指数大于0表示该特征与目标变量整体呈正向关系，小于0表示整体呈负向关系。"
    )
    summary_lines.append(
        "- 灰色关联度刻画的是变化趋势接近程度，不代表严格因果；Spearman符号用于补充方向解释。"
    )
    return "\n".join(summary_lines) + "\n"


def main() -> None:
    raw_file = PREPROCESS_DIR / "预处理后数据_未标准化.csv"
    raw_df = pd.read_csv(raw_file, encoding="utf-8-sig")

    minmax_df = raw_df.copy()
    for column in FEATURE_COLUMNS + TARGET_COLUMNS:
        minmax_df[column] = min_max_scale(raw_df[column])
    minmax_df.to_csv(OUTPUT_DIR / "gra_minmax_standardized_data.csv", index=False, encoding="utf-8-sig")

    dflow_result, dflow_coeff = gra_for_target(minmax_df, raw_df, "DFlow")
    dgap_result, dgap_coeff = gra_for_target(minmax_df, raw_df, "DGap")

    dflow_result.to_csv(OUTPUT_DIR / "DFlow_gra_results.csv", index=False, encoding="utf-8-sig")
    dgap_result.to_csv(OUTPUT_DIR / "DGap_gra_results.csv", index=False, encoding="utf-8-sig")
    dflow_coeff.to_csv(OUTPUT_DIR / "DFlow_gra_coefficients_by_sample.csv", index=False, encoding="utf-8-sig")
    dgap_coeff.to_csv(OUTPUT_DIR / "DGap_gra_coefficients_by_sample.csv", index=False, encoding="utf-8-sig")

    all_results = pd.concat(
        [
            dflow_result.assign(target="DFlow"),
            dgap_result.assign(target="DGap"),
        ],
        ignore_index=True,
    )
    all_results.to_csv(OUTPUT_DIR / "Q1_gra_combined_results.csv", index=False, encoding="utf-8-sig")

    plot_gra_bar(dflow_result, "DFlow")
    plot_gra_bar(dgap_result, "DGap")
    plot_signed_influence(dflow_result, "DFlow")
    plot_signed_influence(dgap_result, "DGap")
    plot_coefficient_heatmap(dflow_coeff, "DFlow")
    plot_coefficient_heatmap(dgap_coeff, "DGap")
    plot_target_comparison(dflow_result, dgap_result)

    summary = build_summary(dflow_result, dgap_result, raw_file)
    (OUTPUT_DIR / "Q1_GRA_summary.md").write_text(summary, encoding="utf-8")


if __name__ == "__main__":
    main()
