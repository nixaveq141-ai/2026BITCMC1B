from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent.parent
OUTPUT_DIR = SCRIPT_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

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
RHO_VALUES = [0.3, 0.5, 0.7]


def find_preprocessed_file() -> Path:
    candidates = [
        ROOT_DIR / "预处理后数据_未标准化.csv",
        ROOT_DIR / "数据预处理" / "output" / "预处理后数据_未标准化.csv",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError("未找到预处理后数据_未标准化.csv")


def min_max_scale(series: pd.Series) -> pd.Series:
    span = series.max() - series.min()
    if span == 0:
        return pd.Series(0.0, index=series.index)
    return (series - series.min()) / span


def gra_rank_for_target(minmax_df: pd.DataFrame, target: str, rho: float) -> pd.DataFrame:
    reference = minmax_df[target]
    compare = minmax_df[FEATURE_COLUMNS]
    delta = compare.sub(reference, axis=0).abs()
    delta_min = delta.min().min()
    delta_max = delta.max().max()
    coeff = (delta_min + rho * delta_max) / (delta + rho * delta_max)
    gra_degree = coeff.mean(axis=0)
    result = pd.DataFrame(
        {
            "feature": FEATURE_COLUMNS,
            "gra_degree": [gra_degree[col] for col in FEATURE_COLUMNS],
        }
    )
    result["rank"] = result["gra_degree"].rank(ascending=False, method="min").astype(int)
    result["target"] = target
    result["rho"] = rho
    return result.sort_values(["rank", "feature"]).reset_index(drop=True)


def plot_degree_heatmap(results_df: pd.DataFrame, target: str) -> None:
    pivot = results_df[results_df["target"] == target].pivot(
        index="feature", columns="rho", values="gra_degree"
    ).loc[FEATURE_COLUMNS]
    plt.figure(figsize=(8, 9))
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap="YlGnBu")
    plt.xlabel("rho")
    plt.ylabel("Feature")
    plt.title(f"{target} GRA Degree Sensitivity")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"{target}_rho_degree_sensitivity.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_rank_heatmap(results_df: pd.DataFrame, target: str) -> None:
    pivot = results_df[results_df["target"] == target].pivot(
        index="feature", columns="rho", values="rank"
    ).loc[FEATURE_COLUMNS]
    plt.figure(figsize=(8, 9))
    sns.heatmap(pivot, annot=True, fmt=".0f", cmap="YlOrRd_r")
    plt.xlabel("rho")
    plt.ylabel("Feature")
    plt.title(f"{target} GRA Rank Sensitivity")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"{target}_rho_rank_sensitivity.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_top5_presence(stability_df: pd.DataFrame, target: str) -> None:
    target_df = stability_df[stability_df["target"] == target].copy()
    top_features = target_df.sort_values(
        ["top5_count", "mean_rank"], ascending=[False, True]
    )["feature"].tolist()
    presence_cols = [f"rho_{rho}" for rho in RHO_VALUES]
    heat_df = target_df.set_index("feature")[presence_cols].loc[top_features]
    plt.figure(figsize=(6, 8))
    sns.heatmap(heat_df, annot=True, fmt=".0f", cmap="Greens", cbar=False)
    plt.xlabel("rho in Top-5")
    plt.ylabel("Feature")
    plt.title(f"{target} Top-5 Stability")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"{target}_top5_stability.png", dpi=300, bbox_inches="tight")
    plt.close()


def build_summary(top5_summary: pd.DataFrame, stability_summary: pd.DataFrame) -> str:
    lines = [
        "# GRA 分辨系数敏感性实验摘要",
        "",
        "- 实验目的：改变灰色关联分析分辨系数 rho=0.3、0.5、0.7，观察 DFlow 与 DGap 的前五特征排序是否稳定。",
        "- 判断原则：若某特征在 3 个 rho 取值下均进入前五，则认为该特征具有较强稳定性。",
        "",
    ]
    for target in TARGET_COLUMNS:
        lines.append(f"## {target} 前五特征对比")
        target_top5 = top5_summary[top5_summary["target"] == target]
        for _, row in target_top5.iterrows():
            lines.append(f"- rho={row['rho']:.1f}：{row['top5_features']}")
        stable = stability_summary[
            (stability_summary["target"] == target) & (stability_summary["top5_count"] == len(RHO_VALUES))
        ]["feature"].tolist()
        lines.append(
            f"- 在三个 rho 下均进入前五的稳定特征：{', '.join(stable) if stable else '无'}。"
        )
        lines.append("")
    return "\n".join(lines) + "\n"


def main() -> None:
    data_file = find_preprocessed_file()
    raw_df = pd.read_csv(data_file, encoding="utf-8-sig")
    minmax_df = raw_df.copy()
    for col in FEATURE_COLUMNS + TARGET_COLUMNS:
        minmax_df[col] = min_max_scale(raw_df[col])

    results = []
    top5_rows = []
    for target in TARGET_COLUMNS:
        for rho in RHO_VALUES:
            result_df = gra_rank_for_target(minmax_df, target, rho)
            results.append(result_df)
            top5 = result_df.head(5)["feature"].tolist()
            top5_rows.append(
                {
                    "target": target,
                    "rho": rho,
                    "top5_features": ", ".join(top5),
                }
            )

    all_results = pd.concat(results, ignore_index=True)
    all_results.to_csv(OUTPUT_DIR / "gra_rho_sensitivity_full_results.csv", index=False, encoding="utf-8-sig")

    top5_summary = pd.DataFrame(top5_rows)
    top5_summary.to_csv(OUTPUT_DIR / "gra_rho_top5_summary.csv", index=False, encoding="utf-8-sig")

    stability_rows = []
    for target in TARGET_COLUMNS:
        target_df = all_results[all_results["target"] == target]
        for feature in FEATURE_COLUMNS:
            row = {"target": target, "feature": feature}
            top5_count = 0
            rank_values = []
            for rho in RHO_VALUES:
                rank = int(target_df[(target_df["feature"] == feature) & (target_df["rho"] == rho)]["rank"].iloc[0])
                row[f"rank_{rho}"] = rank
                in_top5 = int(rank <= 5)
                row[f"rho_{rho}"] = in_top5
                top5_count += in_top5
                rank_values.append(rank)
            row["top5_count"] = top5_count
            row["mean_rank"] = sum(rank_values) / len(rank_values)
            stability_rows.append(row)

    stability_summary = pd.DataFrame(stability_rows).sort_values(
        ["target", "top5_count", "mean_rank"], ascending=[True, False, True]
    )
    stability_summary.to_csv(OUTPUT_DIR / "gra_rho_top5_stability_summary.csv", index=False, encoding="utf-8-sig")

    for target in TARGET_COLUMNS:
        plot_degree_heatmap(all_results, target)
        plot_rank_heatmap(all_results, target)
        plot_top5_presence(stability_summary, target)

    summary = build_summary(top5_summary, stability_summary)
    with open(OUTPUT_DIR / "GRA_sensitivity_summary.md", "w", encoding="utf-8-sig") as f:
        f.write(summary)

    print("GRA sensitivity analysis completed.")


if __name__ == "__main__":
    main()
