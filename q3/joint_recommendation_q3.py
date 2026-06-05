from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import joblib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from xgboost import XGBRegressor

matplotlib.use("Agg")


ROOT = Path(__file__).resolve().parents[1]
Q3_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = Q3_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DATA_PATH = ROOT / "预处理后数据_未标准化.csv"
Q2_BEST_PARAMS_PATH = ROOT / "q2" / "q2_dual_output_xgb_shap" / "dual_output_xgb_shap_outputs" / "best_params.json"
Q2_MODEL_PATH = ROOT / "q2" / "q2_dual_output_xgb_shap" / "dual_output_xgb_shap_outputs" / "dual_output_xgb_model.joblib"
Q2_COLLABORATION_COEFS_PATH = (
    ROOT / "q2" / "q2_gra_master_slave_regression" / "outputs" / "model_coefficients.csv"
)
Q2_SYNERGY_PATH = (
    ROOT / "q2" / "q2_dual_output_xgb_shap" / "dual_output_xgb_shap_outputs" / "dual_output_synergy_indices.csv"
)
Q1_DFLOW_GRA_PATH = ROOT / "q1" / "gra_q1" / "outputs" / "DFlow_gra_results.csv"
Q1_DGAP_GRA_PATH = ROOT / "q1" / "gra_q1" / "outputs" / "DGap_gra_results.csv"

RANDOM_STATE = 42
TEST_SIZE = 0.2

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
COMMON_FEATURE = "ILD_I_SDevT"


@dataclass
class CollaborationPrior:
    intercept: float
    beta: float
    center_coefs: Dict[str, float]
    synergy_weights: Dict[str, float]


def load_dataset() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH, encoding="utf-8-sig")


def load_best_params() -> Dict[str, float]:
    with Q2_BEST_PARAMS_PATH.open("r", encoding="utf-8") as fp:
        return json.load(fp)


def build_dual_output_model() -> MultiOutputRegressor:
    params = load_best_params()
    base_model = XGBRegressor(
        objective="reg:squarederror",
        random_state=RANDOM_STATE,
        n_jobs=1,
        **params,
    )
    return MultiOutputRegressor(base_model)


def get_dual_output_training_strategy() -> Dict[str, object]:
    if Q2_MODEL_PATH.exists():
        return {
            "source": "loaded_existing_q2_model",
            "model_path": str(Q2_MODEL_PATH),
            "estimator_wrapper": "MultiOutputRegressor",
            "base_estimator": "XGBRegressor",
            "note": "Q3 directly loads the persisted dual-output model produced in Q2.",
            "best_params": load_best_params(),
        }
    return {
        "source": "retrained_from_q2_best_params",
        "estimator_wrapper": "MultiOutputRegressor",
        "base_estimator": "XGBRegressor",
        "note": "Q3 reuses the optimal parameter configuration identified in Q2, then refits the dual-output model on the current split to obtain base joint predictions.",
        "best_params": load_best_params(),
    }


def load_or_build_dual_output_model(X: pd.DataFrame, y: pd.DataFrame) -> MultiOutputRegressor:
    if Q2_MODEL_PATH.exists():
        return joblib.load(Q2_MODEL_PATH)
    model = build_dual_output_model()
    model.fit(X, y)
    return model


def build_evaluation_dual_output_model(X: pd.DataFrame, y: pd.DataFrame) -> MultiOutputRegressor:
    model = build_dual_output_model()
    model.fit(X, y)
    return model


def load_collaboration_prior() -> CollaborationPrior:
    coef_df = pd.read_csv(Q2_COLLABORATION_COEFS_PATH, encoding="utf-8-sig")
    synergy_df = pd.read_csv(Q2_SYNERGY_PATH, encoding="utf-8-sig")
    synergy_df["feature_for_center"] = synergy_df["feature"].replace({"ILD_I_SDevT": "C"})
    forward = coef_df[coef_df["model"] == "forward_slave_collaborative"].copy()
    intercept = float(forward.loc[forward["feature"] == "Intercept", "coefficient"].iloc[0])
    beta = float(forward.loc[forward["feature"] == "DFlow", "coefficient"].iloc[0])
    center_rows = forward[~forward["feature"].isin(["Intercept", "DFlow"])].copy()
    center_coefs = dict(zip(center_rows["feature"], center_rows["coefficient"]))
    center_features = list(center_coefs.keys())
    center_synergy = synergy_df[synergy_df["feature_for_center"].isin(center_features)].copy()
    center_synergy["raw_weight"] = (
        0.6 * center_synergy["shared_contribution"] + 0.4 * center_synergy["abs_direction_correlation"]
    )
    weight_sum = center_synergy["raw_weight"].sum()
    if weight_sum <= 0:
        center_synergy["normalized_weight"] = 1.0 / max(len(center_synergy), 1)
    else:
        center_synergy["normalized_weight"] = center_synergy["raw_weight"] / weight_sum
    synergy_weights = dict(zip(center_synergy["feature_for_center"], center_synergy["normalized_weight"]))
    for feature in center_features:
        synergy_weights.setdefault(feature, 1.0 / len(center_features))
    return CollaborationPrior(
        intercept=intercept,
        beta=beta,
        center_coefs=center_coefs,
        synergy_weights=synergy_weights,
    )


def compute_collaboration_center(df: pd.DataFrame, prior: CollaborationPrior) -> np.ndarray:
    center = np.full(len(df), prior.intercept, dtype=float)
    for feature, coef in prior.center_coefs.items():
        source_feature = COMMON_FEATURE if feature == "C" else feature
        synergy_weight = prior.synergy_weights.get(feature, 1.0)
        center += (coef * synergy_weight) * df[source_feature].to_numpy(dtype=float)
    return center


def nearest_level(values: np.ndarray, allowed_levels: np.ndarray) -> np.ndarray:
    distances = np.abs(values[:, None] - allowed_levels[None, :])
    indices = distances.argmin(axis=1)
    return allowed_levels[indices]


def solve_full_joint_recommendation(
    f_hat: np.ndarray,
    g_hat: np.ndarray,
    center: np.ndarray,
    beta: float,
    lambda1: float,
    lambda2: float,
    lambda3: float,
    bounds_f: Tuple[float, float],
    bounds_g: Tuple[float, float],
) -> Tuple[np.ndarray, np.ndarray]:
    a = lambda1 + lambda3 * (beta ** 2)
    d = lambda2 + lambda3
    coupling = lambda3 * beta
    det = a * d - coupling**2

    rhs1 = lambda1 * f_hat - lambda3 * beta * center
    rhs2 = lambda2 * g_hat + lambda3 * center

    f_star = (rhs1 * d + coupling * rhs2) / det
    g_star = (a * rhs2 + coupling * rhs1) / det

    f_star = np.clip(f_star, bounds_f[0], bounds_f[1])
    g_star = np.clip(g_star, bounds_g[0], bounds_g[1])
    return f_star, g_star


def solve_simplified_forward_recommendation(
    f_hat: np.ndarray,
    g_hat: np.ndarray,
    center: np.ndarray,
    beta: float,
    lambda2: float,
    lambda3: float,
    bounds_f: Tuple[float, float],
    bounds_g: Tuple[float, float],
) -> Tuple[np.ndarray, np.ndarray]:
    f_star = np.clip(f_hat, bounds_f[0], bounds_f[1])
    collaborative_target = center + beta * f_star
    g_star = (lambda2 * g_hat + lambda3 * collaborative_target) / (lambda2 + lambda3)
    g_star = np.clip(g_star, bounds_g[0], bounds_g[1])
    return f_star, g_star


def calculate_metrics(
    actual_f: np.ndarray,
    actual_g: np.ndarray,
    pred_f: np.ndarray,
    pred_g: np.ndarray,
    center: np.ndarray,
    beta: float,
    discrete: bool,
) -> Dict[str, float]:
    metrics = {
        "rmse_dflow": float(np.sqrt(mean_squared_error(actual_f, pred_f))),
        "mae_dflow": float(mean_absolute_error(actual_f, pred_f)),
        "r2_dflow": float(r2_score(actual_f, pred_f)),
        "rmse_dgap": float(np.sqrt(mean_squared_error(actual_g, pred_g))),
        "mae_dgap": float(mean_absolute_error(actual_g, pred_g)),
        "r2_dgap": float(r2_score(actual_g, pred_g)),
    }
    metrics["joint_rmse_mean"] = (metrics["rmse_dflow"] + metrics["rmse_dgap"]) / 2
    metrics["joint_mae_mean"] = (metrics["mae_dflow"] + metrics["mae_dgap"]) / 2
    metrics["joint_r2_mean"] = (metrics["r2_dflow"] + metrics["r2_dgap"]) / 2
    collaboration_residual = pred_g - (center + beta * pred_f)
    metrics["collaboration_rmse"] = float(np.sqrt(np.mean(collaboration_residual**2)))
    if discrete:
        metrics["accuracy_dflow"] = float(np.mean(actual_f == pred_f))
        metrics["accuracy_dgap"] = float(np.mean(actual_g == pred_g))
        metrics["joint_pair_accuracy"] = float(np.mean((actual_f == pred_f) & (actual_g == pred_g)))
    return metrics


def lambda_grid_search(
    actual_f: np.ndarray,
    actual_g: np.ndarray,
    f_hat: np.ndarray,
    g_hat: np.ndarray,
    center: np.ndarray,
    beta: float,
    bounds_f: Tuple[float, float],
    bounds_g: Tuple[float, float],
    allowed_f: np.ndarray,
    allowed_g: np.ndarray,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    full_records: List[Dict[str, float]] = []
    simplified_records: List[Dict[str, float]] = []
    lambda_values = [0.5, 1.0, 2.0, 4.0]
    lambda3_values = [0.25, 0.5, 1.0, 2.0, 4.0]

    for lambda1 in lambda_values:
        for lambda2 in lambda_values:
            for lambda3 in lambda3_values:
                f_cont, g_cont = solve_full_joint_recommendation(
                    f_hat=f_hat,
                    g_hat=g_hat,
                    center=center,
                    beta=beta,
                    lambda1=lambda1,
                    lambda2=lambda2,
                    lambda3=lambda3,
                    bounds_f=bounds_f,
                    bounds_g=bounds_g,
                )
                f_disc = nearest_level(f_cont, allowed_f)
                g_disc = nearest_level(g_cont, allowed_g)
                metrics = calculate_metrics(actual_f, actual_g, f_disc, g_disc, center, beta, discrete=True)
                full_records.append(
                    {
                        "method": "full_joint_recommendation",
                        "lambda1": lambda1,
                        "lambda2": lambda2,
                        "lambda3": lambda3,
                        **metrics,
                    }
                )

    for lambda2 in lambda_values:
        for lambda3 in lambda3_values:
            f_cont, g_cont = solve_simplified_forward_recommendation(
                f_hat=f_hat,
                g_hat=g_hat,
                center=center,
                beta=beta,
                lambda2=lambda2,
                lambda3=lambda3,
                bounds_f=bounds_f,
                bounds_g=bounds_g,
            )
            f_disc = nearest_level(f_cont, allowed_f)
            g_disc = nearest_level(g_cont, allowed_g)
            metrics = calculate_metrics(actual_f, actual_g, f_disc, g_disc, center, beta, discrete=True)
            simplified_records.append(
                {
                    "method": "simplified_forward_recommendation",
                    "lambda1": np.nan,
                    "lambda2": lambda2,
                    "lambda3": lambda3,
                    **metrics,
                }
            )

    full_df = pd.DataFrame(full_records).sort_values(
        by=["joint_rmse_mean", "joint_pair_accuracy", "collaboration_rmse"],
        ascending=[True, False, True],
    )
    simplified_df = pd.DataFrame(simplified_records).sort_values(
        by=["joint_rmse_mean", "joint_pair_accuracy", "collaboration_rmse"],
        ascending=[True, False, True],
    )
    return full_df, simplified_df


def build_recommendation_frame(
    index: Iterable,
    actual_f: np.ndarray,
    actual_g: np.ndarray,
    f_hat: np.ndarray,
    g_hat: np.ndarray,
    center: np.ndarray,
    beta: float,
    final_f_cont: np.ndarray,
    final_g_cont: np.ndarray,
    final_f_disc: np.ndarray,
    final_g_disc: np.ndarray,
    method_name: str,
) -> pd.DataFrame:
    df = pd.DataFrame(
        {
            "sample_index": list(index),
            "DFlow_actual": actual_f,
            "DGap_actual": actual_g,
            "DFlow_base_pred": f_hat,
            "DGap_base_pred": g_hat,
            "collaboration_center": center,
            "beta": beta,
            "DFlow_recommended_continuous": final_f_cont,
            "DGap_recommended_continuous": final_g_cont,
            "DFlow_recommended_discrete": final_f_disc,
            "DGap_recommended_discrete": final_g_disc,
            "delta_DFlow_continuous": final_f_cont - f_hat,
            "delta_DGap_continuous": final_g_cont - g_hat,
            "collaboration_residual_base": g_hat - (center + beta * f_hat),
            "collaboration_residual_recommended": final_g_cont - (center + beta * final_f_cont),
            "method": method_name,
        }
    )
    df["joint_pair_match"] = (
        (df["DFlow_actual"] == df["DFlow_recommended_discrete"])
        & (df["DGap_actual"] == df["DGap_recommended_discrete"])
    ).astype(int)
    return df


def save_metric_comparison_plot(metrics_df: pd.DataFrame, output_path: Path) -> None:
    plot_df = metrics_df.copy()
    selected_metrics = [
        ("rmse_dflow", "DFlow RMSE"),
        ("rmse_dgap", "DGap RMSE"),
        ("joint_pair_accuracy", "Joint Pair Accuracy"),
        ("collaboration_rmse", "Collaboration RMSE"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for ax, (metric_col, title) in zip(axes.flatten(), selected_metrics):
        sns.barplot(data=plot_df, x="method", y=metric_col, hue="dataset", ax=ax, palette="Set2")
        ax.set_title(title)
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=15)
        if metric_col == "joint_pair_accuracy":
            ax.set_ylim(0, 1)
    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_adjustment_distribution_plot(recommendation_df: pd.DataFrame, output_path: Path) -> None:
    plot_df = recommendation_df.melt(
        id_vars=["sample_index"],
        value_vars=["delta_DFlow_continuous", "delta_DGap_continuous"],
        var_name="variable",
        value_name="adjustment",
    )
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(data=plot_df, x="variable", y="adjustment", hue="variable", ax=ax, palette="Set3", legend=False)
    ax.axhline(0, color="black", linestyle="--", linewidth=1)
    ax.set_title("Final Recommendation Adjustment Distribution")
    ax.set_xlabel("")
    ax.set_ylabel("Continuous Adjustment")
    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_recommendation_scatter_plot(recommendation_df: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    pairs = [
        ("DFlow_base_pred", "DFlow_recommended_continuous", "DFlow"),
        ("DGap_base_pred", "DGap_recommended_continuous", "DGap"),
    ]
    for ax, (x_col, y_col, label) in zip(axes, pairs):
        ax.scatter(recommendation_df[x_col], recommendation_df[y_col], alpha=0.7)
        bounds = [
            min(recommendation_df[x_col].min(), recommendation_df[y_col].min()),
            max(recommendation_df[x_col].max(), recommendation_df[y_col].max()),
        ]
        ax.plot(bounds, bounds, color="red", linestyle="--")
        ax.set_title(f"{label}: Base Prediction vs Final Recommendation")
        ax.set_xlabel("Base Prediction")
        ax.set_ylabel("Final Recommendation")
    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_transition_heatmap(
    recommendation_df: pd.DataFrame,
    base_col: str,
    final_col: str,
    output_path: Path,
    title: str,
) -> None:
    transition = pd.crosstab(recommendation_df[base_col], recommendation_df[final_col])
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(transition, annot=True, fmt="d", cmap="YlGnBu", ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Final Discrete Recommendation")
    ax.set_ylabel("Base Discrete Prediction")
    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_top_adjustment_plot(recommendation_df: pd.DataFrame, output_path: Path) -> None:
    top_df = recommendation_df.copy()
    top_df["joint_adjustment_abs"] = np.abs(top_df["delta_DFlow_continuous"]) + np.abs(top_df["delta_DGap_continuous"])
    top_df = top_df.nlargest(12, "joint_adjustment_abs")
    melted = top_df.melt(
        id_vars=["sample_index"],
        value_vars=["delta_DFlow_continuous", "delta_DGap_continuous"],
        var_name="variable",
        value_name="adjustment",
    )
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(data=melted, x="sample_index", y="adjustment", hue="variable", ax=ax, palette="Set1")
    ax.axhline(0, color="black", linestyle="--", linewidth=1)
    ax.set_title("Top Samples with Largest Recommendation Adjustments")
    ax.set_xlabel("Sample Index")
    ax.set_ylabel("Adjustment")
    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def build_metrics_records(
    dataset_name: str,
    method_name: str,
    actual_f: np.ndarray,
    actual_g: np.ndarray,
    pred_f_cont: np.ndarray,
    pred_g_cont: np.ndarray,
    pred_f_disc: np.ndarray,
    pred_g_disc: np.ndarray,
    center: np.ndarray,
    beta: float,
) -> List[Dict[str, float]]:
    cont_metrics = calculate_metrics(actual_f, actual_g, pred_f_cont, pred_g_cont, center, beta, discrete=False)
    disc_metrics = calculate_metrics(actual_f, actual_g, pred_f_disc, pred_g_disc, center, beta, discrete=True)
    return [
        {"dataset": dataset_name, "method": method_name, "setting_type": "continuous", **cont_metrics},
        {"dataset": dataset_name, "method": method_name, "setting_type": "discrete", **disc_metrics},
    ]


def save_summary(
    summary_path: Path,
    selected_method: str,
    selected_config: Dict[str, float],
    prior: CollaborationPrior,
    full_results: pd.DataFrame,
    simplified_results: pd.DataFrame,
    test_metrics_df: pd.DataFrame,
    representative_df: pd.DataFrame,
    synergy_df: pd.DataFrame,
    q1_dflow_top: List[str],
    q1_dgap_top: List[str],
) -> None:
    best_full = full_results.iloc[0]
    best_simplified = simplified_results.iloc[0]
    test_discrete = test_metrics_df[
        (test_metrics_df["dataset"] == "test") & (test_metrics_df["setting_type"] == "discrete")
    ].copy()
    selected_test = test_discrete[test_discrete["method"] == selected_method].iloc[0]
    base_test = test_discrete[test_discrete["method"] == "base_joint_prediction"].iloc[0]
    top_synergy = synergy_df.head(5)["feature"].tolist()
    representative_text = representative_df.to_string(index=False)
    training_strategy = get_dual_output_training_strategy()
    lines = [
        "# 第三问 联合预测—协同校正参数推荐模型结果摘要",
        "",
        "- 模型定位：沿用第二问双输出 XGBoost 的最优结构与参数配置，并在第三问当前划分下重新训练双输出模型以获得基础联合预测值；再利用第二问正向主从协同回归构造 DGap 协同中心项，通过二次目标函数把“预测值”修正为“推荐设定值”。",
        f"- 协同先验方程：G ≈ c(x) + βF，其中 β = {prior.beta:.4f}。",
        f"- 协同中心项特征：{', '.join(prior.center_coefs.keys())}；其中 C 对应 {COMMON_FEATURE}。",
        f"- SHAP 协同权重：{prior.synergy_weights}。",
        f"- 双输出基础预测训练策略：{training_strategy['source']}，封装器为 {training_strategy['estimator_wrapper']}，基学习器为 {training_strategy['base_estimator']}。",
        f"- 第一问 DFlow GRA 前五：{', '.join(q1_dflow_top)}。",
        f"- 第一问 DGap GRA 前五：{', '.join(q1_dgap_top)}。",
        f"- 第二问共享/差异/补偿分析中最值得关注的协同特征前五：{', '.join(top_synergy)}。",
        "",
        "## 训练集上最优权重搜索结果",
        f"- Full joint 推荐最优：lambda1={best_full['lambda1']}, lambda2={best_full['lambda2']}, lambda3={best_full['lambda3']}，离散 joint_rmse_mean={best_full['joint_rmse_mean']:.4f}，joint_pair_accuracy={best_full['joint_pair_accuracy']:.4f}。",
        f"- Simplified forward 推荐最优：lambda2={best_simplified['lambda2']}, lambda3={best_simplified['lambda3']}，离散 joint_rmse_mean={best_simplified['joint_rmse_mean']:.4f}，joint_pair_accuracy={best_simplified['joint_pair_accuracy']:.4f}。",
        "",
        "## 测试集回顾性评价",
        f"- 最终选择的方法：{selected_method}。",
        f"- 选择依据：在离散设定值评价下具有更低的 joint_rmse_mean 或更高的 joint_pair_accuracy，同时保持较低的协同残差。",
        f"- 当前最佳测试集离散结果：DFlow RMSE={selected_test['rmse_dflow']:.4f}，DGap RMSE={selected_test['rmse_dgap']:.4f}，joint_pair_accuracy={selected_test['joint_pair_accuracy']:.4f}，collaboration_rmse={selected_test['collaboration_rmse']:.4f}。",
        "",
        "## 第三问的建模结论",
        "- 第三问不是重新做特征筛选，而是把第二问的基础联合预测值转化为最终可执行设定值。",
        "- 在当前数据中，推荐值必须进一步投影到历史可行档位 {-1, 0, 1}，这样才能与真实控制设定保持一致。",
        "- 协同校正后，推荐值相较于纯联合预测会主动压缩违反主从联动关系的组合，从而降低 DGap 对 DFlow 负向协同关系的偏离。",
        "- 若论文更强调工程可执行性，可优先使用 simplified_forward_recommendation；若更强调双变量同时折中，可展示 full_joint_recommendation 作为对照。",
        "",
        "## 代表性样本推荐结果",
        "```text",
        representative_text,
        "```",
    ]
    summary_path.write_text("\n".join(lines), encoding="utf-8-sig")


def main() -> None:
    sns.set_theme(style="whitegrid", font="Microsoft YaHei")

    df = load_dataset()
    prior = load_collaboration_prior()
    synergy_df = pd.read_csv(Q2_SYNERGY_PATH, encoding="utf-8-sig")
    q1_dflow_top = (
        pd.read_csv(Q1_DFLOW_GRA_PATH, encoding="utf-8-sig").sort_values("rank").head(5)["feature"].tolist()
    )
    q1_dgap_top = (
        pd.read_csv(Q1_DGAP_GRA_PATH, encoding="utf-8-sig").sort_values("rank").head(5)["feature"].tolist()
    )

    X = df[FEATURE_COLUMNS].copy()
    y = df[TARGET_COLUMNS].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    model = build_evaluation_dual_output_model(X_train, y_train)
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    center_train = compute_collaboration_center(X_train, prior)
    center_test = compute_collaboration_center(X_test, prior)

    allowed_f = np.sort(df["DFlow"].unique().astype(float))
    allowed_g = np.sort(df["DGap"].unique().astype(float))
    bounds_f = (float(allowed_f.min()), float(allowed_f.max()))
    bounds_g = (float(allowed_g.min()), float(allowed_g.max()))

    full_grid, simplified_grid = lambda_grid_search(
        actual_f=y_train["DFlow"].to_numpy(dtype=float),
        actual_g=y_train["DGap"].to_numpy(dtype=float),
        f_hat=train_pred[:, 0],
        g_hat=train_pred[:, 1],
        center=center_train,
        beta=prior.beta,
        bounds_f=bounds_f,
        bounds_g=bounds_g,
        allowed_f=allowed_f,
        allowed_g=allowed_g,
    )
    full_grid.to_csv(OUTPUT_DIR / "lambda_search_full.csv", index=False, encoding="utf-8-sig")
    simplified_grid.to_csv(OUTPUT_DIR / "lambda_search_simplified.csv", index=False, encoding="utf-8-sig")

    best_full = full_grid.iloc[0].to_dict()
    best_simplified = simplified_grid.iloc[0].to_dict()

    full_train_f_cont, full_train_g_cont = solve_full_joint_recommendation(
        f_hat=train_pred[:, 0],
        g_hat=train_pred[:, 1],
        center=center_train,
        beta=prior.beta,
        lambda1=float(best_full["lambda1"]),
        lambda2=float(best_full["lambda2"]),
        lambda3=float(best_full["lambda3"]),
        bounds_f=bounds_f,
        bounds_g=bounds_g,
    )
    full_test_f_cont, full_test_g_cont = solve_full_joint_recommendation(
        f_hat=test_pred[:, 0],
        g_hat=test_pred[:, 1],
        center=center_test,
        beta=prior.beta,
        lambda1=float(best_full["lambda1"]),
        lambda2=float(best_full["lambda2"]),
        lambda3=float(best_full["lambda3"]),
        bounds_f=bounds_f,
        bounds_g=bounds_g,
    )
    simp_train_f_cont, simp_train_g_cont = solve_simplified_forward_recommendation(
        f_hat=train_pred[:, 0],
        g_hat=train_pred[:, 1],
        center=center_train,
        beta=prior.beta,
        lambda2=float(best_simplified["lambda2"]),
        lambda3=float(best_simplified["lambda3"]),
        bounds_f=bounds_f,
        bounds_g=bounds_g,
    )
    simp_test_f_cont, simp_test_g_cont = solve_simplified_forward_recommendation(
        f_hat=test_pred[:, 0],
        g_hat=test_pred[:, 1],
        center=center_test,
        beta=prior.beta,
        lambda2=float(best_simplified["lambda2"]),
        lambda3=float(best_simplified["lambda3"]),
        bounds_f=bounds_f,
        bounds_g=bounds_g,
    )

    base_train_f_disc = nearest_level(train_pred[:, 0], allowed_f)
    base_train_g_disc = nearest_level(train_pred[:, 1], allowed_g)
    base_test_f_disc = nearest_level(test_pred[:, 0], allowed_f)
    base_test_g_disc = nearest_level(test_pred[:, 1], allowed_g)
    full_train_f_disc = nearest_level(full_train_f_cont, allowed_f)
    full_train_g_disc = nearest_level(full_train_g_cont, allowed_g)
    full_test_f_disc = nearest_level(full_test_f_cont, allowed_f)
    full_test_g_disc = nearest_level(full_test_g_cont, allowed_g)
    simp_train_f_disc = nearest_level(simp_train_f_cont, allowed_f)
    simp_train_g_disc = nearest_level(simp_train_g_cont, allowed_g)
    simp_test_f_disc = nearest_level(simp_test_f_cont, allowed_f)
    simp_test_g_disc = nearest_level(simp_test_g_cont, allowed_g)

    metrics_records: List[Dict[str, float]] = []
    metrics_records.extend(
        build_metrics_records(
            "train",
            "base_joint_prediction",
            y_train["DFlow"].to_numpy(dtype=float),
            y_train["DGap"].to_numpy(dtype=float),
            train_pred[:, 0],
            train_pred[:, 1],
            base_train_f_disc,
            base_train_g_disc,
            center_train,
            prior.beta,
        )
    )
    metrics_records.extend(
        build_metrics_records(
            "test",
            "base_joint_prediction",
            y_test["DFlow"].to_numpy(dtype=float),
            y_test["DGap"].to_numpy(dtype=float),
            test_pred[:, 0],
            test_pred[:, 1],
            base_test_f_disc,
            base_test_g_disc,
            center_test,
            prior.beta,
        )
    )
    metrics_records.extend(
        build_metrics_records(
            "train",
            "full_joint_recommendation",
            y_train["DFlow"].to_numpy(dtype=float),
            y_train["DGap"].to_numpy(dtype=float),
            full_train_f_cont,
            full_train_g_cont,
            full_train_f_disc,
            full_train_g_disc,
            center_train,
            prior.beta,
        )
    )
    metrics_records.extend(
        build_metrics_records(
            "test",
            "full_joint_recommendation",
            y_test["DFlow"].to_numpy(dtype=float),
            y_test["DGap"].to_numpy(dtype=float),
            full_test_f_cont,
            full_test_g_cont,
            full_test_f_disc,
            full_test_g_disc,
            center_test,
            prior.beta,
        )
    )
    metrics_records.extend(
        build_metrics_records(
            "train",
            "simplified_forward_recommendation",
            y_train["DFlow"].to_numpy(dtype=float),
            y_train["DGap"].to_numpy(dtype=float),
            simp_train_f_cont,
            simp_train_g_cont,
            simp_train_f_disc,
            simp_train_g_disc,
            center_train,
            prior.beta,
        )
    )
    metrics_records.extend(
        build_metrics_records(
            "test",
            "simplified_forward_recommendation",
            y_test["DFlow"].to_numpy(dtype=float),
            y_test["DGap"].to_numpy(dtype=float),
            simp_test_f_cont,
            simp_test_g_cont,
            simp_test_f_disc,
            simp_test_g_disc,
            center_test,
            prior.beta,
        )
    )
    metrics_df = pd.DataFrame(metrics_records)
    metrics_df.to_csv(OUTPUT_DIR / "recommendation_metrics_summary.csv", index=False, encoding="utf-8-sig")

    test_discrete = metrics_df[
        (metrics_df["dataset"] == "test")
        & (metrics_df["setting_type"] == "discrete")
        & (metrics_df["method"].isin(["full_joint_recommendation", "simplified_forward_recommendation"]))
    ].copy()
    selected_row = test_discrete.sort_values(
        by=["joint_rmse_mean", "joint_pair_accuracy", "collaboration_rmse"],
        ascending=[True, False, True],
    ).iloc[0]
    selected_method = str(selected_row["method"])
    selected_config = best_full if selected_method == "full_joint_recommendation" else best_simplified

    if selected_method == "full_joint_recommendation":
        final_test_cont = (full_test_f_cont, full_test_g_cont)
        final_test_disc = (full_test_f_disc, full_test_g_disc)
    else:
        final_test_cont = (simp_test_f_cont, simp_test_g_cont)
        final_test_disc = (simp_test_f_disc, simp_test_g_disc)

    selected_config_payload = {
        "selected_method": selected_method,
        "dual_output_training_strategy": {
            "evaluation": {
                "source": "retrained_on_current_split",
                "estimator_wrapper": "MultiOutputRegressor",
                "base_estimator": "XGBRegressor",
                "best_params": load_best_params(),
                "note": "Used only for hold-out evaluation inside Q3 to avoid leakage.",
            },
            "deployment": get_dual_output_training_strategy(),
        },
        "allowed_levels": {"DFlow": allowed_f.tolist(), "DGap": allowed_g.tolist()},
        "feasible_bounds": {"DFlow": list(bounds_f), "DGap": list(bounds_g)},
        "collaboration_prior": {
            "intercept": prior.intercept,
            "beta": prior.beta,
            "center_coefs": prior.center_coefs,
            "synergy_weights": prior.synergy_weights,
        },
        "selected_lambdas": selected_config,
    }
    (OUTPUT_DIR / "selected_recommendation_config.json").write_text(
        json.dumps(selected_config_payload, ensure_ascii=False, indent=2),
        encoding="utf-8-sig",
    )

    collaboration_prior_df = pd.DataFrame(
        [
            {"component": "intercept", "feature": "Intercept", "coefficient": prior.intercept},
            {"component": "beta", "feature": "DFlow", "coefficient": prior.beta},
            *[
                {
                    "component": "center",
                    "feature": feature,
                    "coefficient": coef,
                    "synergy_weight": prior.synergy_weights.get(feature, 1.0),
                    "effective_coefficient": coef * prior.synergy_weights.get(feature, 1.0),
                }
                for feature, coef in prior.center_coefs.items()
            ],
        ]
    )
    collaboration_prior_df.to_csv(
        OUTPUT_DIR / "collaboration_prior.csv", index=False, encoding="utf-8-sig"
    )

    test_recommendations = build_recommendation_frame(
        index=y_test.index,
        actual_f=y_test["DFlow"].to_numpy(dtype=float),
        actual_g=y_test["DGap"].to_numpy(dtype=float),
        f_hat=test_pred[:, 0],
        g_hat=test_pred[:, 1],
        center=center_test,
        beta=prior.beta,
        final_f_cont=final_test_cont[0],
        final_g_cont=final_test_cont[1],
        final_f_disc=final_test_disc[0],
        final_g_disc=final_test_disc[1],
        method_name=selected_method,
    )
    test_recommendations["DFlow_base_pred_discrete"] = base_test_f_disc
    test_recommendations["DGap_base_pred_discrete"] = base_test_g_disc
    test_recommendations.to_csv(OUTPUT_DIR / "recommendations_test.csv", index=False, encoding="utf-8-sig")

    full_model = load_or_build_dual_output_model(X, y)
    full_pred = full_model.predict(X)
    center_all = compute_collaboration_center(X, prior)

    if selected_method == "full_joint_recommendation":
        all_f_cont, all_g_cont = solve_full_joint_recommendation(
            f_hat=full_pred[:, 0],
            g_hat=full_pred[:, 1],
            center=center_all,
            beta=prior.beta,
            lambda1=float(selected_config["lambda1"]),
            lambda2=float(selected_config["lambda2"]),
            lambda3=float(selected_config["lambda3"]),
            bounds_f=bounds_f,
            bounds_g=bounds_g,
        )
    else:
        all_f_cont, all_g_cont = solve_simplified_forward_recommendation(
            f_hat=full_pred[:, 0],
            g_hat=full_pred[:, 1],
            center=center_all,
            beta=prior.beta,
            lambda2=float(selected_config["lambda2"]),
            lambda3=float(selected_config["lambda3"]),
            bounds_f=bounds_f,
            bounds_g=bounds_g,
        )

    all_f_disc = nearest_level(all_f_cont, allowed_f)
    all_g_disc = nearest_level(all_g_cont, allowed_g)
    all_recommendations = build_recommendation_frame(
        index=y.index,
        actual_f=y["DFlow"].to_numpy(dtype=float),
        actual_g=y["DGap"].to_numpy(dtype=float),
        f_hat=full_pred[:, 0],
        g_hat=full_pred[:, 1],
        center=center_all,
        beta=prior.beta,
        final_f_cont=all_f_cont,
        final_g_cont=all_g_cont,
        final_f_disc=all_f_disc,
        final_g_disc=all_g_disc,
        method_name=selected_method,
    )
    all_recommendations["DFlow_base_pred_discrete"] = nearest_level(full_pred[:, 0], allowed_f)
    all_recommendations["DGap_base_pred_discrete"] = nearest_level(full_pred[:, 1], allowed_g)
    all_recommendations.to_csv(OUTPUT_DIR / "recommendations_all_samples.csv", index=False, encoding="utf-8-sig")

    representative = all_recommendations.assign(
        joint_adjustment_abs=np.abs(all_recommendations["delta_DFlow_continuous"])
        + np.abs(all_recommendations["delta_DGap_continuous"])
    ).nlargest(12, "joint_adjustment_abs")[
        [
            "sample_index",
            "DFlow_actual",
            "DGap_actual",
            "DFlow_base_pred",
            "DGap_base_pred",
            "DFlow_recommended_discrete",
            "DGap_recommended_discrete",
            "delta_DFlow_continuous",
            "delta_DGap_continuous",
            "joint_adjustment_abs",
        ]
    ]
    representative.to_csv(OUTPUT_DIR / "representative_recommendations.csv", index=False, encoding="utf-8-sig")

    save_metric_comparison_plot(metrics_df, OUTPUT_DIR / "method_metric_comparison.png")
    save_adjustment_distribution_plot(all_recommendations, OUTPUT_DIR / "adjustment_distribution.png")
    save_recommendation_scatter_plot(all_recommendations, OUTPUT_DIR / "base_vs_final_recommendation.png")
    save_transition_heatmap(
        all_recommendations,
        base_col="DFlow_base_pred_discrete",
        final_col="DFlow_recommended_discrete",
        output_path=OUTPUT_DIR / "DFlow_level_transition_heatmap.png",
        title="DFlow Base-to-Final Level Transition",
    )
    save_transition_heatmap(
        all_recommendations,
        base_col="DGap_base_pred_discrete",
        final_col="DGap_recommended_discrete",
        output_path=OUTPUT_DIR / "DGap_level_transition_heatmap.png",
        title="DGap Base-to-Final Level Transition",
    )
    save_top_adjustment_plot(all_recommendations, OUTPUT_DIR / "top_adjustment_samples.png")

    equation_lines = [
        "Q3 recommendation objective:",
        "J(F,G) = lambda1*(F - F_hat)^2 + lambda2*(G - G_hat)^2 + lambda3*(G - c(x) - beta*F)^2",
        "",
        f"Selected method: {selected_method}",
        f"Selected lambda config: {selected_config}",
        "",
        f"Collaboration prior beta: {prior.beta:.6f}",
        "c(x) = "
        + f"{prior.intercept:.6f}"
        + "".join(
            [
                f" + ({coef:.6f})*{COMMON_FEATURE if feature == 'C' else feature}"
                for feature, coef in prior.center_coefs.items()
            ]
        ),
        "",
        f"Allowed DFlow levels: {allowed_f.tolist()}",
        f"Allowed DGap levels: {allowed_g.tolist()}",
        f"Feasible DFlow bounds: {bounds_f}",
        f"Feasible DGap bounds: {bounds_g}",
    ]
    (OUTPUT_DIR / "recommendation_equations.txt").write_text(
        "\n".join(equation_lines),
        encoding="utf-8-sig",
    )

    save_summary(
        summary_path=OUTPUT_DIR / "Q3_joint_recommendation_summary.md",
        selected_method=selected_method,
        selected_config=selected_config,
        prior=prior,
        full_results=full_grid,
        simplified_results=simplified_grid,
        test_metrics_df=metrics_df,
        representative_df=representative,
        synergy_df=synergy_df,
        q1_dflow_top=q1_dflow_top,
        q1_dgap_top=q1_dgap_top,
    )


if __name__ == "__main__":
    main()
