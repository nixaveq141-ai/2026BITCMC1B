from __future__ import annotations

from pathlib import Path
from zipfile import ZipFile
import xml.etree.ElementTree as ET

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "预处理结果"
OUTPUT_DIR.mkdir(exist_ok=True)

plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial"]
plt.rcParams["axes.unicode_minus"] = False
sns.set_theme(style="whitegrid")


def find_file(pattern: str) -> Path:
    matches = list(BASE_DIR.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"未找到匹配文件: {pattern}")
    return matches[0]


def extract_docx_text(docx_path: Path) -> str:
    with ZipFile(docx_path) as zf:
        root = ET.fromstring(zf.read("word/document.xml"))
    ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    paragraphs: list[str] = []
    for para in root.findall(".//w:p", ns):
        texts = [node.text for node in para.findall(".//w:t", ns) if node.text]
        if texts:
            paragraphs.append("".join(texts))
    return "\n".join(paragraphs)


def main() -> None:
    docx_path = find_file("*一级赛题*.docx")
    csv_path = find_file("*202601B*.csv")

    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    original_df = df.copy()

    all_columns = df.columns.tolist()
    target_cols = ["DFlow", "DGap"]
    feature_cols = [col for col in all_columns if col not in target_cols]

    missing_summary = pd.DataFrame(
        {
            "column": all_columns,
            "missing_count": df.isna().sum().values,
            "missing_ratio": (df.isna().mean().round(6)).values,
            "handling_method": [
                "无需填补（原始数据无缺失）" if df[col].isna().sum() == 0 else "待填补"
                for col in all_columns
            ],
            "reason": [
                "该列未发现缺失值，保留原始观测。"
                if df[col].isna().sum() == 0
                else "若后续新增缺失，时间序列优先线性插值，非时序连续变量优先KNN填补，低比例离散缺失可用众数或均值。"
                for col in all_columns
            ],
        }
    )
    missing_summary.to_csv(OUTPUT_DIR / "缺失值检测结果.csv", index=False, encoding="utf-8-sig")

    duplicate_rows = int(df.duplicated().sum())
    duplicate_report = pd.DataFrame(
        [
            {
                "row_count": len(df),
                "duplicate_row_count": duplicate_rows,
                "handling": "保留重复记录",
                "reason": "电火花加工数据可能包含重复工况下的重复试验，删除会损失频次信息，不利于后续建模。",
            }
        ]
    )
    duplicate_report.to_csv(OUTPUT_DIR / "重复值说明.csv", index=False, encoding="utf-8-sig")

    outlier_records: list[dict[str, float | str | int]] = []
    clipped_df = df.copy()

    for col in feature_cols:
        series = df[col]
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        mask = (series < lower) | (series > upper)
        flagged = df.loc[mask, col]

        for idx, value in flagged.items():
            outlier_records.append(
                {
                    "row_index": int(idx),
                    "column": col,
                    "original_value": float(value),
                    "lower_bound": round(float(lower), 6),
                    "upper_bound": round(float(upper), 6),
                    "treatment": "IQR截尾修正",
                    "reason": "样本量较小且极值可能是真实工况，采用分位边界截尾，避免直接删样本。",
                }
            )

        clipped_df[col] = series.clip(lower=lower, upper=upper)

    outlier_df = pd.DataFrame(outlier_records)
    if outlier_df.empty:
        outlier_df = pd.DataFrame(
            columns=[
                "row_index",
                "column",
                "original_value",
                "lower_bound",
                "upper_bound",
                "treatment",
                "reason",
            ]
        )
    outlier_df.to_csv(OUTPUT_DIR / "异常值明细_IQR.csv", index=False, encoding="utf-8-sig")

    before_count = (
        outlier_df.groupby("column").size().reindex(feature_cols, fill_value=0).rename("outlier_count_before")
    )

    after_records: list[dict[str, float | str | int]] = []
    for col in feature_cols:
        series = clipped_df[col]
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        mask = (series < lower) | (series > upper)
        for idx, value in clipped_df.loc[mask, col].items():
            after_records.append({"row_index": int(idx), "column": col, "value": float(value)})

    after_count = (
        pd.DataFrame(after_records).groupby("column").size().reindex(feature_cols, fill_value=0).rename("outlier_count_after")
        if after_records
        else pd.Series(0, index=feature_cols, name="outlier_count_after")
    )

    bounds_rows = []
    for col in feature_cols:
        series = df[col]
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        bounds_rows.append(
            {
                "column": col,
                "Q1": round(float(q1), 6),
                "Q3": round(float(q3), 6),
                "IQR": round(float(iqr), 6),
                "lower_bound": round(float(lower), 6),
                "upper_bound": round(float(upper), 6),
                "outlier_count_before": int(before_count[col]),
                "outlier_count_after": int(after_count[col]),
                "handling": "IQR检测 + 分位边界截尾",
            }
        )

    pd.DataFrame(bounds_rows).to_csv(OUTPUT_DIR / "异常值统计汇总.csv", index=False, encoding="utf-8-sig")

    flagged_cols = [row["column"] for row in bounds_rows if row["outlier_count_before"] > 0]
    if not flagged_cols:
        flagged_cols = feature_cols[:4]

    fig, axes = plt.subplots(2, 1, figsize=(16, 10), constrained_layout=True)
    sns.boxplot(data=original_df[flagged_cols], orient="h", ax=axes[0], color="#e6a23c")
    axes[0].set_title("Outlier Boxplot Before Treatment (IQR)")
    axes[0].set_xlabel("Value")
    axes[0].set_ylabel("Feature")

    sns.boxplot(data=clipped_df[flagged_cols], orient="h", ax=axes[1], color="#67c23a")
    axes[1].set_title("Outlier Boxplot After Winsorization")
    axes[1].set_xlabel("Value")
    axes[1].set_ylabel("Feature")
    fig.savefig(OUTPUT_DIR / "异常值箱线图.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(clipped_df[feature_cols])
    scaled_df = pd.DataFrame(scaled_features, columns=feature_cols)
    scaled_df[target_cols] = clipped_df[target_cols].values

    scaler_params = pd.DataFrame(
        {
            "column": feature_cols,
            "mean_u": scaler.mean_.round(6),
            "std_sigma": scaler.scale_.round(6),
            "formula": ["z = (x - u) / sigma"] * len(feature_cols),
            "need_scaling": ["是"] * len(feature_cols),
            "reason": [
                "脉冲特征量纲不同，标准化有利于后续回归、SVM、聚类和距离度量类模型稳定训练。"
            ]
            * len(feature_cols),
        }
    )
    scaler_params.to_csv(OUTPUT_DIR / "标准化参数.csv", index=False, encoding="utf-8-sig")

    corr_df = scaled_df[feature_cols + target_cols].corr()
    plt.figure(figsize=(16, 12))
    sns.heatmap(corr_df, cmap="RdBu_r", center=0, annot=False, square=False)
    plt.title("Correlation Heatmap After Standardization")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "相关性热力图.png", dpi=300, bbox_inches="tight")
    plt.close()

    clipped_df.to_csv(OUTPUT_DIR / "预处理后数据_未标准化.csv", index=False, encoding="utf-8-sig")
    scaled_df.to_csv(OUTPUT_DIR / "预处理后数据_标准化.csv", index=False, encoding="utf-8-sig")

    supplementary_df = pd.DataFrame(
        [
            {
                "supplement_type": "必须补充",
                "name": "加工质量约束等级",
                "symbol": "QualityLevel",
                "assumed_value": "Q2",
                "unit": "等级",
                "basis": "题目要求“确保加工质量”但原始数据未给出质量标签，预处理阶段需先设定一个基准质量等级用于后续约束建模。",
                "recommended_source": "企业历史工艺卡、设备出厂手册、实验室质检记录",
            },
            {
                "supplement_type": "必须补充",
                "name": "表面粗糙度上限",
                "symbol": "Ra_max",
                "assumed_value": 1.6,
                "unit": "μm",
                "basis": "中等精度电火花加工常以Ra不高于1.6μm作为较保守的工艺质量门槛，便于筛选可行控制区间。",
                "recommended_source": "设备说明书、加工工艺手册、企业质检标准",
            },
            {
                "supplement_type": "必须补充",
                "name": "电极损耗率上限",
                "symbol": "EWR_max",
                "assumed_value": 0.12,
                "unit": "比例",
                "basis": "控制变量优化若不约束电极损耗，可能出现效率提升但成本失控；0.12作为保守假设便于后续筛选。",
                "recommended_source": "企业历史台账、设备维护记录、工艺实验数据",
            },
            {
                "supplement_type": "可选补充",
                "name": "材料去除率目标",
                "symbol": "MRR_target",
                "assumed_value": 0.85,
                "unit": "相对指数",
                "basis": "当前数据仅含脉冲特征和控制量，没有效率标签，可引入相对去除率目标辅助多目标优化。",
                "recommended_source": "企业试验数据、论文附录、Kaggle/学校公开实验数据",
            },
            {
                "supplement_type": "可选补充",
                "name": "工作液温度",
                "symbol": "FluidTemp",
                "assumed_value": 25.0,
                "unit": "℃",
                "basis": "温度会影响放电稳定性与流量效果，若现场无法实时采集，可先假设恒温25℃进行基准分析。",
                "recommended_source": "设备传感器记录、车间MES系统、人工巡检记录",
            },
        ]
    )
    supplementary_df.to_csv(OUTPUT_DIR / "补充数据建议与假设值.csv", index=False, encoding="utf-8-sig")

    assumed_sample_df = clipped_df.copy()
    stability_cols = ["ASD_A_SDevT", "BSD_B_SDevT", "CSD_C_SDevT", "ISD_I_SDevT", "ALD_A_SDevT", "BLD_B_SDevT", "CLD_C_SDevT", "ILD_I_SDevT"]
    denom = assumed_sample_df[stability_cols].sum(axis=1).replace(0, 1e-6)
    assumed_sample_df["QualityProxy"] = (1 / denom).round(6)
    assumed_sample_df["CoordinationCost"] = ((assumed_sample_df["DFlow"].abs() + assumed_sample_df["DGap"].abs()) / 2).round(6)
    assumed_sample_df["FluidTemp"] = 25.0
    assumed_sample_df["RaConstraint"] = 1.6
    assumed_sample_df["EWRConstraint"] = 0.12
    assumed_sample_df.to_csv(OUTPUT_DIR / "补充数据_基于假设的样本级指标.csv", index=False, encoding="utf-8-sig")

    report = f"""# 数据预处理说明

## 1. 数据概况
- 题目文件：{docx_path.name}
- 原始数据文件：{csv_path.name}
- 原始样本量：{len(original_df)} 行
- 原始字段数：{len(all_columns)} 列
- 其中脉冲特征：16 列
- 控制变量：2 列（DFlow, DGap）
- 重复记录：{duplicate_rows} 行，已保留

## 2. 缺失值处理
- 检测结果：当前数据集各字段缺失值均为 0。
- 本次处理：不进行强制填补，直接保留原始观测。
- 方案依据：
  - 若后续补采形成时间序列缺失，应优先使用线性插值；
  - 若为非时序连续变量且存在多变量相关性，可优先使用 KNN 填充；
  - 若缺失比例极低且变量较稳定，可使用均值填充作为基线方法。

## 3. 异常值处理
- 检测方法：IQR 法。
- 判定公式：
  - 下界 = Q1 - 1.5 × IQR
  - 上界 = Q3 + 1.5 × IQR
  - IQR = Q3 - Q1
- 处理原则：对 16 个连续脉冲特征进行检测；对识别出的异常值不删样本，而采用分位边界截尾修正。
- 处理依据：样本量仅 154 条，直接删除会削弱样本代表性；同时电火花加工中的极端脉冲状态可能对应真实工况，因此更适合温和修正。
- 图表：`异常值箱线图.png`

## 4. 转换处理
- 是否需要标准化：需要。
- 原因：16 个脉冲特征的数值范围差异明显，后续若建立回归、SVM、聚类、主成分或距离度量模型，标准化可以避免大尺度变量主导模型。
- 标准化公式：z = (x - u) / sigma
  - 其中 u 为样本均值，sigma 为样本标准差。
- 处理说明：仅对 16 个脉冲特征标准化，DFlow 与 DGap 作为控制变量保留原始离散取值。

## 5. 数据补充建议
- 必须补充：
  - 加工质量标签或质量约束（如表面粗糙度、尺寸误差、电极损耗率），因为原始数据无法直接支撑“确保加工质量”这一约束。
- 可选补充：
  - 材料去除率、工作液温度、工件材质、脉冲频率等，有助于提升模型精度与泛化能力。
- 推荐来源：
  - 国家统计局官网：适合获取宏观制造业背景参数；
  - 企业历史工艺卡、设备说明书、MES/质检系统：优先级最高；
  - Kaggle、论文附录、学校实验平台：适合作为补充验证数据。
- 本次基于合理假设生成了两类补充结果：
  - `补充数据建议与假设值.csv`：给出必须/可选补充项及假设依据；
  - `补充数据_基于假设的样本级指标.csv`：在样本层面增加 QualityProxy、CoordinationCost、FluidTemp 等辅助变量，供后续建模敏感性分析使用。

## 6. 结果文件说明
- `预处理后数据_未标准化.csv`：完成异常值修正后的数据。
- `预处理后数据_标准化.csv`：在上一步基础上，对 16 个脉冲特征做标准化后的数据。
- `缺失值检测结果.csv`：各字段缺失值统计与处理说明。
- `异常值明细_IQR.csv`：异常值样本明细。
- `异常值统计汇总.csv`：各字段 IQR 阈值与异常值计数。
- `标准化参数.csv`：标准化均值、标准差与公式。
- `相关性热力图.png`：标准化后变量相关性图。

## 7. 题目文本摘要
{extract_docx_text(docx_path)}
"""
    (OUTPUT_DIR / "预处理说明.md").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
