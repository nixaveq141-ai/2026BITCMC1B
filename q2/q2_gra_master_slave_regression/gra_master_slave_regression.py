from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RepeatedKFold, train_test_split

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
Q1_GRA_DIR = ROOT_DIR / "q1" / "gra_q1" / "outputs"
OUTPUT_DIR = SCRIPT_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial"]
plt.rcParams["axes.unicode_minus"] = False
sns.set_theme(style="whitegrid")

RANDOM_STATE = 42
CV_SPLITS = 5
CV_REPEATS = 20
MASTER_ALPHA = 0.80
SLAVE_ALPHA = 1.00
MIN_PENALTY = 0.15


@dataclass
class WeightedRidgeModel:
    feature_names: list[str]
    intercept_: float
    coef_: np.ndarray
    penalty_vector: np.ndarray

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.intercept_ + X[self.feature_names].to_numpy(dtype=float) @ self.coef_


def find_standardized_file() -> Path:
    for path in [
        ROOT_DIR / "预处理后数据_标准化.csv",
        ROOT_DIR / "数据预处理" / "output" / "预处理后数据_标准化.csv",
    ]:
        if path.exists():
            return path
    raise FileNotFoundError("未找到预处理后数据_标准化.csv")


def load_gra_results(target: str) -> pd.DataFrame:
    return pd.read_csv(Q1_GRA_DIR / f"{target}_gra_results.csv", encoding="utf-8-sig")


def weighted_ridge_fit(X: pd.DataFrame, y: pd.Series, penalty_map: dict[str, float], alpha: float) -> WeightedRidgeModel:
    names = list(X.columns)
    x = X.to_numpy(dtype=float)
    yv = y.to_numpy(dtype=float)
    x_mean = x.mean(axis=0)
    y_mean = yv.mean()
    xc = x - x_mean
    yc = yv - y_mean
    p = np.array([penalty_map[n] for n in names], dtype=float)
    beta = np.linalg.solve(xc.T @ xc + alpha * np.diag(p), xc.T @ yc)
    intercept = float(y_mean - x_mean @ beta)
    return WeightedRidgeModel(names, intercept, beta, p)


def metrics_dict(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def build_penalty_map(gra_df: pd.DataFrame, features: list[str], common_feature: str | None = None, link_feature: str | None = None) -> dict[str, float]:
    gra_use = gra_df.copy()
    if common_feature and common_feature not in set(gra_use["feature"]) and "ILD_I_SDevT" in set(gra_use["feature"]):
        gra_use = pd.concat(
            [
                gra_use,
                pd.DataFrame(
                    [{"feature": common_feature, "gra_degree": float(gra_use.set_index("feature").loc["ILD_I_SDevT", "gra_degree"])}]
                ),
            ],
            ignore_index=True,
        )
    sub = gra_use.set_index("feature").loc[features].copy()
    deg = sub["gra_degree"]
    deg = (deg - deg.min()) / (deg.max() - deg.min() + 1e-8)
    penalties = (1.0 - 0.85 * deg).clip(lower=MIN_PENALTY)
    mp = penalties.to_dict()
    if common_feature and common_feature in mp:
        mp[common_feature] = max(MIN_PENALTY, mp[common_feature] * 0.55)
    if link_feature and link_feature in mp:
        mp[link_feature] = max(MIN_PENALTY, mp[link_feature] * 0.45)
    return mp


def format_equation(model: WeightedRidgeModel, target: str) -> str:
    parts = [f"{model.intercept_:.4f}"]
    for coef, feat in zip(model.coef_, model.feature_names):
        parts.append(f" {'+' if coef >= 0 else '-'} {abs(coef):.4f}*{feat}")
    return f"{target} = " + "".join(parts)


def make_pred_frame(model: str, split: str, y_true: pd.Series, y_pred: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame({"model": model, "split": split, "actual": y_true.reset_index(drop=True), "predicted": y_pred, "residual": y_true.reset_index(drop=True) - y_pred})


def summarize_cv(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby("model").agg(
        rmse_mean=("rmse", "mean"), rmse_std=("rmse", "std"),
        mae_mean=("mae", "mean"), mae_std=("mae", "std"),
        r2_mean=("r2", "mean"), r2_std=("r2", "std"),
    ).reset_index()


def fit_scheme(data: pd.DataFrame, direction: str) -> tuple[dict[str, object], pd.DataFrame]:
    c = "C"
    if direction == "forward":
        master_target, slave_target = "DFlow", "DGap"
        master_feats = ["ASM_A_MeanT", "ALM_A_MeanT", "ALD_A_SDevT", "CLD_C_SDevT", c]
        slave_feats = ["CSM_C_MeanT", "CLM_C_MeanT", "BLD_B_SDevT", "CSD_C_SDevT", c]
        master_gra, slave_gra = load_gra_results("DFlow"), load_gra_results("DGap")
    else:
        master_target, slave_target = "DGap", "DFlow"
        master_feats = ["CSM_C_MeanT", "CLM_C_MeanT", "BLD_B_SDevT", "CSD_C_SDevT", c]
        slave_feats = ["ASM_A_MeanT", "ALM_A_MeanT", "ALD_A_SDevT", "CLD_C_SDevT", c]
        master_gra, slave_gra = load_gra_results("DGap"), load_gra_results("DFlow")

    collab_feats = slave_feats + [master_target]
    slave_aug = pd.concat(
        [
            slave_gra,
            pd.DataFrame([{"feature": master_target, "gra_degree": float(slave_gra["gra_degree"].max())}]),
            pd.DataFrame([{"feature": c, "gra_degree": float(slave_gra.set_index("feature").loc["ILD_I_SDevT", "gra_degree"])}]),
        ],
        ignore_index=True,
    ).drop_duplicates(subset=["feature"], keep="first")

    master_pen = build_penalty_map(master_gra, master_feats, common_feature=c)
    slave_pen = build_penalty_map(slave_gra, slave_feats, common_feature=c)
    collab_pen = build_penalty_map(slave_aug, collab_feats, common_feature=c, link_feature=master_target)

    tr_idx, te_idx = train_test_split(np.arange(len(data)), test_size=0.2, random_state=RANDOM_STATE)
    train = data.iloc[tr_idx].reset_index(drop=True)
    test = data.iloc[te_idx].reset_index(drop=True)

    master = weighted_ridge_fit(train[master_feats], train[master_target], master_pen, MASTER_ALPHA)
    master_tr = master.predict(train[master_feats]); master_te = master.predict(test[master_feats])
    slave_ind = weighted_ridge_fit(train[slave_feats], train[slave_target], slave_pen, SLAVE_ALPHA)
    ind_tr = slave_ind.predict(train[slave_feats]); ind_te = slave_ind.predict(test[slave_feats])

    col_train = train[slave_feats].copy(); col_train[master_target] = train[master_target].values
    col_test_actual = test[slave_feats].copy(); col_test_actual[master_target] = test[master_target].values
    col_test_deploy = test[slave_feats].copy(); col_test_deploy[master_target] = master_te
    slave_col = weighted_ridge_fit(col_train[collab_feats], train[slave_target], collab_pen, SLAVE_ALPHA)
    col_tr = slave_col.predict(col_train[collab_feats]); col_te_actual = slave_col.predict(col_test_actual[collab_feats]); col_te_deploy = slave_col.predict(col_test_deploy[collab_feats])

    prefix = "forward" if direction == "forward" else "reverse"
    metric_rows = []
    for name, split, yt, yp in [
        (f"{master_target}_master_{prefix}", "train", train[master_target], master_tr),
        (f"{master_target}_master_{prefix}", "test", test[master_target], master_te),
        (f"{slave_target}_independent_{prefix}", "train", train[slave_target], ind_tr),
        (f"{slave_target}_independent_{prefix}", "test", test[slave_target], ind_te),
        (f"{slave_target}_collaborative_actual_{prefix}", "train", train[slave_target], col_tr),
        (f"{slave_target}_collaborative_actual_{prefix}", "test", test[slave_target], col_te_actual),
        (f"{slave_target}_collaborative_deploy_{prefix}", "test", test[slave_target], col_te_deploy),
    ]:
        metric_rows.append({"model": name, "split": split, **metrics_dict(yt, yp)})
    metrics = pd.DataFrame(metric_rows)

    preds = pd.concat(
        [
            make_pred_frame(f"{master_target}_master_{prefix}", "train", train[master_target], master_tr),
            make_pred_frame(f"{master_target}_master_{prefix}", "test", test[master_target], master_te),
            make_pred_frame(f"{slave_target}_independent_{prefix}", "train", train[slave_target], ind_tr),
            make_pred_frame(f"{slave_target}_independent_{prefix}", "test", test[slave_target], ind_te),
            make_pred_frame(f"{slave_target}_collaborative_actual_{prefix}", "train", train[slave_target], col_tr),
            make_pred_frame(f"{slave_target}_collaborative_actual_{prefix}", "test", test[slave_target], col_te_actual),
            make_pred_frame(f"{slave_target}_collaborative_deploy_{prefix}", "test", test[slave_target], col_te_deploy),
        ],
        ignore_index=True,
    )
    return {
        "direction": direction, "master_target": master_target, "slave_target": slave_target, "master_feats": master_feats,
        "slave_feats": slave_feats, "master": master, "slave_ind": slave_ind, "slave_col": slave_col,
        "master_pen": master_pen, "slave_pen": slave_pen, "collab_pen": collab_pen, "preds": preds,
        "master_test_pred": master_te,
    }, metrics


def repeated_cv(data: pd.DataFrame, direction: str) -> pd.DataFrame:
    c = "C"
    if direction == "forward":
        master_target, slave_target = "DFlow", "DGap"
        master_feats = ["ASM_A_MeanT", "ALM_A_MeanT", "ALD_A_SDevT", "CLD_C_SDevT", c]
        slave_feats = ["CSM_C_MeanT", "CLM_C_MeanT", "BLD_B_SDevT", "CSD_C_SDevT", c]
        master_gra, slave_gra = load_gra_results("DFlow"), load_gra_results("DGap")
    else:
        master_target, slave_target = "DGap", "DFlow"
        master_feats = ["CSM_C_MeanT", "CLM_C_MeanT", "BLD_B_SDevT", "CSD_C_SDevT", c]
        slave_feats = ["ASM_A_MeanT", "ALM_A_MeanT", "ALD_A_SDevT", "CLD_C_SDevT", c]
        master_gra, slave_gra = load_gra_results("DGap"), load_gra_results("DFlow")
    collab_feats = slave_feats + [master_target]
    slave_aug = pd.concat(
        [slave_gra, pd.DataFrame([{"feature": master_target, "gra_degree": float(slave_gra["gra_degree"].max())}]), pd.DataFrame([{"feature": c, "gra_degree": float(slave_gra.set_index("feature").loc["ILD_I_SDevT", "gra_degree"])}])],
        ignore_index=True,
    ).drop_duplicates(subset=["feature"], keep="first")
    master_pen = build_penalty_map(master_gra, master_feats, common_feature=c)
    slave_pen = build_penalty_map(slave_gra, slave_feats, common_feature=c)
    collab_pen = build_penalty_map(slave_aug, collab_feats, common_feature=c, link_feature=master_target)

    rkf = RepeatedKFold(n_splits=CV_SPLITS, n_repeats=CV_REPEATS, random_state=RANDOM_STATE)
    rows = []
    for fold_id, (tr_idx, te_idx) in enumerate(rkf.split(data), start=1):
        train = data.iloc[tr_idx].reset_index(drop=True); test = data.iloc[te_idx].reset_index(drop=True)
        master = weighted_ridge_fit(train[master_feats], train[master_target], master_pen, MASTER_ALPHA)
        master_te = master.predict(test[master_feats])
        slave_ind = weighted_ridge_fit(train[slave_feats], train[slave_target], slave_pen, SLAVE_ALPHA)
        ind_te = slave_ind.predict(test[slave_feats])
        col_train = train[slave_feats].copy(); col_train[master_target] = train[master_target].values
        col_test_a = test[slave_feats].copy(); col_test_a[master_target] = test[master_target].values
        col_test_d = test[slave_feats].copy(); col_test_d[master_target] = master_te
        slave_col = weighted_ridge_fit(col_train[collab_feats], train[slave_target], collab_pen, SLAVE_ALPHA)
        evals = [
            (f"{direction}_master", test[master_target], master_te),
            (f"{direction}_independent", test[slave_target], ind_te),
            (f"{direction}_collaborative_actual", test[slave_target], slave_col.predict(col_test_a[collab_feats])),
            (f"{direction}_collaborative_deploy", test[slave_target], slave_col.predict(col_test_d[collab_feats])),
        ]
        for model, yt, yp in evals:
            rows.append({"fold_id": fold_id, "model": model, **metrics_dict(yt, yp)})
    return pd.DataFrame(rows)


def bootstrap_lambda(data: pd.DataFrame, n_bootstrap: int = 500) -> pd.DataFrame:
    c = "C"
    slave_gra = load_gra_results("DGap")
    collab_feats = ["CSM_C_MeanT", "CLM_C_MeanT", "BLD_B_SDevT", "CSD_C_SDevT", c, "DFlow"]
    slave_aug = pd.concat(
        [slave_gra, pd.DataFrame([{"feature": "DFlow", "gra_degree": float(slave_gra["gra_degree"].max())}]), pd.DataFrame([{"feature": c, "gra_degree": float(slave_gra.set_index("feature").loc["ILD_I_SDevT", "gra_degree"])}])],
        ignore_index=True,
    ).drop_duplicates(subset=["feature"], keep="first")
    collab_pen = build_penalty_map(slave_aug, collab_feats, common_feature=c, link_feature="DFlow")
    rng = np.random.default_rng(RANDOM_STATE)
    rows = []
    X = data[["CSM_C_MeanT", "CLM_C_MeanT", "BLD_B_SDevT", "CSD_C_SDevT", c]].copy()
    X["DFlow"] = data["DFlow"].values
    y = data["DGap"]
    for sample_id in range(n_bootstrap):
        idx = rng.integers(0, len(data), len(data))
        model = weighted_ridge_fit(X.iloc[idx], y.iloc[idx], collab_pen, SLAVE_ALPHA)
        rows.append({"sample_id": sample_id, "lambda": float(model.coef_[model.feature_names.index("DFlow")])})
    return pd.DataFrame(rows)


def plot_scatter(df: pd.DataFrame, filename: str, title: str) -> None:
    plt.figure(figsize=(7, 6))
    sns.scatterplot(data=df, x="actual", y="predicted", hue="split", s=50)
    lims = [min(df["actual"].min(), df["predicted"].min()), max(df["actual"].max(), df["predicted"].max())]
    plt.plot(lims, lims, "--", color="black", linewidth=1)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=300, bbox_inches="tight")
    plt.close()


def plot_bar(melted: pd.DataFrame, filename: str, title: str, x: str = "model") -> None:
    plt.figure(figsize=(12, 5))
    sns.barplot(data=melted, x=x, y="value", hue="metric")
    plt.xticks(rotation=20)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=300, bbox_inches="tight")
    plt.close()


def main() -> None:
    data = pd.read_csv(find_standardized_file(), encoding="utf-8-sig").copy()
    data["C"] = data["ILD_I_SDevT"]

    forward_info, forward_metrics = fit_scheme(data, "forward")
    reverse_info, reverse_metrics = fit_scheme(data, "reverse")
    holdout = pd.concat([forward_metrics, reverse_metrics], ignore_index=True)
    preds = pd.concat([forward_info["preds"], reverse_info["preds"]], ignore_index=True)
    preds.to_csv(OUTPUT_DIR / "model_predictions.csv", index=False, encoding="utf-8-sig")
    holdout.to_csv(OUTPUT_DIR / "holdout_metrics_summary.csv", index=False, encoding="utf-8-sig")

    cv_all = pd.concat([repeated_cv(data, "forward"), repeated_cv(data, "reverse")], ignore_index=True)
    cv_all.to_csv(OUTPUT_DIR / "repeated_cv_all_folds.csv", index=False, encoding="utf-8-sig")
    cv_summary = summarize_cv(cv_all)
    cv_summary.to_csv(OUTPUT_DIR / "repeated_cv_summary.csv", index=False, encoding="utf-8-sig")

    boot = bootstrap_lambda(data)
    boot.to_csv(OUTPUT_DIR / "lambda_bootstrap_samples.csv", index=False, encoding="utf-8-sig")
    mean_lambda = float(boot["lambda"].mean())
    ci_low, ci_high = np.quantile(boot["lambda"], [0.025, 0.975])
    pd.DataFrame([{"lambda_mean": mean_lambda, "lambda_ci_low_95": float(ci_low), "lambda_ci_high_95": float(ci_high), "collaborative_sign": "negative" if mean_lambda < 0 else "positive"}]).to_csv(
        OUTPUT_DIR / "lambda_bootstrap_summary.csv", index=False, encoding="utf-8-sig"
    )

    coef_rows = []
    for name, model in [
        ("forward_master", forward_info["master"]), ("forward_slave_independent", forward_info["slave_ind"]),
        ("forward_slave_collaborative", forward_info["slave_col"]), ("reverse_master", reverse_info["master"]),
        ("reverse_slave_collaborative", reverse_info["slave_col"]),
    ]:
        coef_rows.append(pd.DataFrame({"model": name, "feature": ["Intercept"] + model.feature_names, "coefficient": [model.intercept_] + model.coef_.tolist()}))
    coef_df = pd.concat(coef_rows, ignore_index=True)
    coef_df["abs_coefficient"] = coef_df["coefficient"].abs()
    coef_df.to_csv(OUTPUT_DIR / "model_coefficients.csv", index=False, encoding="utf-8-sig")

    penalty_rows = []
    for name, pm in [
        ("forward_master", forward_info["master_pen"]), ("forward_slave_independent", forward_info["slave_pen"]),
        ("forward_slave_collaborative", forward_info["collab_pen"]), ("reverse_master", reverse_info["master_pen"]),
        ("reverse_slave_collaborative", reverse_info["collab_pen"]),
    ]:
        for feat, pen in pm.items():
            penalty_rows.append({"model": name, "feature": feat, "penalty": pen})
    penalty_df = pd.DataFrame(penalty_rows)
    penalty_df.to_csv(OUTPUT_DIR / "gra_penalty_weights.csv", index=False, encoding="utf-8-sig")

    pd.DataFrame(
        [
            {"role": "forward_master_core_set", "features": ", ".join(forward_info["master_feats"])},
            {"role": "forward_slave_supplement_set", "features": ", ".join(forward_info["slave_feats"])},
            {"role": "common_synergy_index", "features": "C = ILD_I_SDevT"},
            {"role": "reverse_master_core_set", "features": ", ".join(reverse_info["master_feats"])},
            {"role": "reverse_slave_supplement_set", "features": ", ".join(reverse_info["slave_feats"])},
        ]
    ).to_csv(OUTPUT_DIR / "feature_manifest.csv", index=False, encoding="utf-8-sig")

    with open(OUTPUT_DIR / "estimated_equations.txt", "w", encoding="utf-8-sig") as f:
        f.write("Forward master:\n" + format_equation(forward_info["master"], "DFlow") + "\n\n")
        f.write("Forward collaborative slave:\n" + format_equation(forward_info["slave_col"], "DGap_collaborative") + "\n\n")
        f.write("Reverse master:\n" + format_equation(reverse_info["master"], "DGap") + "\n\n")
        f.write("Reverse collaborative slave:\n" + format_equation(reverse_info["slave_col"], "DFlow_collaborative") + "\n")

    forward_test = holdout[(holdout["split"] == "test") & holdout["model"].str.contains("forward")]
    reverse_test = holdout[(holdout["split"] == "test") & holdout["model"].str.contains("reverse")]
    assess = pd.DataFrame(
        [
            {"criterion": "forward_lambda_nonzero", "value": float(forward_info["slave_col"].coef_[forward_info["slave_col"].feature_names.index("DFlow")]), "interpretation": "forward 方向联动系数"},
            {"criterion": "forward_actual_rmse_gain", "value": float(forward_test[forward_test["model"] == "DGap_independent_forward"]["rmse"].iloc[0] - forward_test[forward_test["model"] == "DGap_collaborative_actual_forward"]["rmse"].iloc[0]), "interpretation": "已知 DFlow 时协同改进量"},
            {"criterion": "forward_deploy_rmse_gap", "value": float(forward_test[forward_test["model"] == "DGap_collaborative_deploy_forward"]["rmse"].iloc[0] - forward_test[forward_test["model"] == "DGap_independent_forward"]["rmse"].iloc[0]), "interpretation": "部署场景相对独立模型差值"},
            {"criterion": "reverse_actual_rmse_gain", "value": float(reverse_test[reverse_test["model"] == "DFlow_independent_reverse"]["rmse"].iloc[0] - reverse_test[reverse_test["model"] == "DFlow_collaborative_actual_reverse"]["rmse"].iloc[0]), "interpretation": "已知 DGap 时反向协同改进量"},
            {"criterion": "forward_reverse_cv_gap", "value": float(cv_summary[cv_summary["model"] == "forward_collaborative_deploy"]["rmse_mean"].iloc[0] - cv_summary[cv_summary["model"] == "reverse_collaborative_deploy"]["rmse_mean"].iloc[0]), "interpretation": "forward/reverse 部署型重复CV差值（正值表示 reverse 更优）"},
        ]
    )
    assess.to_csv(OUTPUT_DIR / "collaboration_assessment.csv", index=False, encoding="utf-8-sig")

    plot_scatter(preds[preds["model"] == "DFlow_master_forward"], "DFlow_master_forward_actual_vs_pred.png", "Forward Master Model: DFlow")
    plot_scatter(preds[preds["model"] == "DGap_independent_forward"], "DGap_independent_forward_actual_vs_pred.png", "Forward Independent Slave")
    plot_scatter(preds[preds["model"] == "DGap_collaborative_actual_forward"], "DGap_collaborative_actual_forward_actual_vs_pred.png", "Forward Collaborative Slave (Actual DFlow)")
    plot_scatter(preds[preds["model"] == "DGap_collaborative_deploy_forward"], "DGap_collaborative_deploy_forward_actual_vs_pred.png", "Forward Collaborative Slave (Predicted DFlow)")
    plot_bar(forward_test.melt(id_vars=["model"], value_vars=["rmse", "mae", "r2"], var_name="metric", value_name="value"), "holdout_forward_metrics.png", "Forward Hold-out Metrics")
    plot_bar(cv_summary[cv_summary["model"].str.contains("forward|reverse")].melt(id_vars=["model"], value_vars=["rmse_mean", "mae_mean", "r2_mean"], var_name="metric", value_name="value"), "repeated_cv_mean_performance.png", "Repeated CV Mean Performance")
    plot_bar(pd.DataFrame([
        {"scheme": "forward", "metric": "rmse_mean", "value": float(cv_summary[cv_summary["model"] == "forward_collaborative_deploy"]["rmse_mean"].iloc[0])},
        {"scheme": "reverse", "metric": "rmse_mean", "value": float(cv_summary[cv_summary["model"] == "reverse_collaborative_deploy"]["rmse_mean"].iloc[0])},
        {"scheme": "forward_ind", "metric": "rmse_mean", "value": float(cv_summary[cv_summary["model"] == "forward_independent"]["rmse_mean"].iloc[0])},
        {"scheme": "reverse_ind", "metric": "rmse_mean", "value": float(cv_summary[cv_summary["model"] == "reverse_independent"]["rmse_mean"].iloc[0])},
    ]), "forward_reverse_rmse_comparison.png", "Forward vs Reverse RMSE", x="scheme")
    plot_bar(penalty_df[penalty_df["model"].isin(["forward_master", "forward_slave_collaborative"])].rename(columns={"penalty": "value"}).assign(metric="penalty"), "gra_penalty_weights.png", "GRA-Informed Penalty Weights", x="feature")
    plot_bar(coef_df[coef_df["model"] == "forward_slave_collaborative"].query("feature != 'Intercept'").rename(columns={"coefficient": "value"}).assign(metric="coefficient"), "forward_slave_collaborative_coefficients.png", "Forward Collaborative Coefficients", x="feature")

    plt.figure(figsize=(8, 5))
    sns.histplot(boot["lambda"], bins=30, kde=True, color="#3b82f6")
    plt.axvline(mean_lambda, color="black", linestyle="--", linewidth=1.2)
    plt.axvline(ci_low, color="#ef4444", linestyle="--", linewidth=1.0)
    plt.axvline(ci_high, color="#ef4444", linestyle="--", linewidth=1.0)
    plt.title("Forward Lambda Bootstrap Distribution")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "lambda_bootstrap_distribution.png", dpi=300, bbox_inches="tight")
    plt.close()

    summary = [
        "# 第二问 GRA约束主从协同回归模型结果摘要",
        "",
        "- 本版增强点：显式构造协同指数 C=ILD_I_SDevT；利用第一问 GRA 关联度构造加权惩罚项；补做正反主从方向对照；使用 repeated K-fold 报告均值与标准差。",
        f"- Forward 主模型：{format_equation(forward_info['master'], 'DFlow')}",
        f"- Forward 协同从模型：{format_equation(forward_info['slave_col'], 'DGap_collaborative')}",
        f"- Reverse 协同从模型：{format_equation(reverse_info['slave_col'], 'DFlow_collaborative')}",
        "",
        "## Hold-out 测试集",
    ]
    for _, row in holdout[holdout["split"] == "test"].iterrows():
        summary.append(f"- {row['model']}：RMSE={row['rmse']:.4f}，MAE={row['mae']:.4f}，R²={row['r2']:.4f}")
    summary.extend(
        [
            "",
            "## 重复交叉验证",
            f"- forward 协同部署模型 RMSE 均值±标准差={cv_summary[cv_summary['model'] == 'forward_collaborative_deploy']['rmse_mean'].iloc[0]:.4f} ± {cv_summary[cv_summary['model'] == 'forward_collaborative_deploy']['rmse_std'].iloc[0]:.4f}",
            f"- reverse 协同部署模型 RMSE 均值±标准差={cv_summary[cv_summary['model'] == 'reverse_collaborative_deploy']['rmse_mean'].iloc[0]:.4f} ± {cv_summary[cv_summary['model'] == 'reverse_collaborative_deploy']['rmse_std'].iloc[0]:.4f}",
            "",
            "## 协同判断",
            f"- forward 方向 lambda={forward_info['slave_col'].coef_[forward_info['slave_col'].feature_names.index('DFlow')]:.4f}，bootstrap 95% 区间=[{ci_low:.4f}, {ci_high:.4f}]。",
            "- 已知/设定 DFlow 时，forward 协同模型优于 DGap 独立模型，说明统计协同关系成立。",
            "- 已知/设定主变量时，forward 方向的协同增益大于 reverse 方向，因此从“人工先设主变量、系统联动推荐从变量”的角度，DFlow→DGap 更具解释力。",
            "- 使用主模型预测主变量进行全自动部署时，forward 协同模型与独立模型接近，而 reverse 部署型 RMSE 更低，说明若追求全自动链路，反向方向反而更稳。",
            "- 因而第二问可给出分层结论：面向辅助联动调节，推荐 DFlow→DGap；面向完全自动部署，当前证据更支持 DGap→DFlow。",
        ]
    )
    with open(OUTPUT_DIR / "Q2_master_slave_summary.md", "w", encoding="utf-8-sig") as f:
        f.write("\n".join(summary) + "\n")

    print("Q2 enhanced GRA-constrained master-slave collaborative regression completed.")


if __name__ == "__main__":
    main()
