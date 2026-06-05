# 文件解释
总控文件
dual_output_xgb_shap.py
这是完整可执行脚本。它负责：
读取 预处理后数据_未标准化.csv
用 MultiOutputRegressor(XGBRegressor) 建双输出模型
做参数搜索
生成预测结果
分别对 DFlow 和 DGap 做 SHAP 解释
再构造共享驱动、差异驱动和方向协同指标

best_params.json
这是最终选出来的最优 XGBoost 参数。里面保存了 n_estimators、max_depth、learning_rate 等超参数。它说明双输出模型最终是用哪组参数训练的。

Q2_dual_output_xgb_shap_summary.md
这是这一部分的中文摘要。它概括了：
模型定位
最优参数
测试集表现
共享驱动因子前五
差异驱动因子前五
补偿型特征前五
以及最后的解释性结论

性能结果文件
dual_output_cv_folds.csv
这是参数搜索阶段每组参数在每一折上的交叉验证结果明细。它记录了每个参数组合在不同折上的：
综合平均 RMSE
综合平均 MAE
综合平均 R²
以及 DFlow、DGap 各自的指标。
它适合做“参数调优过程分析”。

dual_output_cv_summary.csv
这是对交叉验证结果的汇总表。它把每组参数在 5 折上的平均表现整理出来，并按 joint_rmse_mean 排序。主要用途是说明“为什么选了这组参数”。

dual_output_metric_summary.csv
这是最终模型在训练集和测试集上的结果表。每一行对应一个输出变量和一个数据划分，包含：
rmse
mae
r2
还有联合输出的平均指标：
joint_rmse_mean
joint_mae_mean
joint_r2_mean
这是最核心的性能表。

dual_output_test_predictions.csv
这是测试集预测结果。里面有：
DFlow_actual
DFlow_predicted
DGap_actual
DGap_predicted
joint_prediction_sum
它适合拿来做单样本对照或误差分析。

特征重要性与协同解释文件
dual_output_feature_importance.csv
这是 XGBoost 内置特征重要性表。它分别给出每个特征对 DFlow 和对 DGap 的树模型重要性。它反映的是“树分裂层面谁更重要”。

dual_output_shap_importance.csv
这是 SHAP 全局重要性表。每个输出变量各有一套 SHAP 排名，主要字段包括：
mean_abs_shap：平均绝对 SHAP 值，衡量总体影响强度
mean_signed_shap：平均带符号 SHAP 值，衡量总体推动方向
shap_rank：SHAP 排名
这张表比 XGBoost 内置重要性更适合做解释。

dual_output_synergy_indices.csv
这是第二问这部分最关键的“协同解释表”。每个特征有 4 个核心指标：
shared_contribution：共享贡献指数，越大说明它同时影响两个输出
differential_contribution：差异贡献指数，越大说明它更偏向其中一个输出
direction_correlation：方向协同系数，正值表示同向作用，负值表示补偿/反向作用
driver_type：根据前面指标自动分的类型，如 shared_driver、differential_driver、compensatory_driver
这张表直接回答“哪些特征支撑协同，哪些特征体现差异，哪些特征具有补偿性”。

综合图表
dual_output_actual_vs_pred.png
这是双输出模型的真实值-预测值散点图。左边是 DFlow，右边是 DGap。点越接近对角线，说明对应输出预测越好。

dual_output_metrics.png
这是测试集上 DFlow 和 DGap 的 RMSE/MAE/R² 对比图。适合放到论文里说明双输出模型的预测性能。

dual_output_feature_importance.png
这是 XGBoost 内置特征重要性图。横轴是特征，颜色区分 DFlow 和 DGap。它能直观看出两输出在“树模型重要性”上的异同。

shared_vs_differential_drivers.png
这张图对每个特征同时展示：
共享贡献指数
差异贡献指数
它最适合回答“这个特征更像共同驱动因子，还是更像差异驱动因子”。

direction_synergy_correlation.png
这张图展示每个特征的方向协同系数。大于 0 说明对两个输出倾向同向作用，小于 0 说明倾向补偿/反向作用。它是解释“为什么主从协同回归里会出现负联动”的重要图。

shared_differential_driver_map.png
这是二维驱动地图。横轴是共享贡献，纵轴是差异贡献，点的颜色和大小还编码了方向协同信息。它相当于把每个特征放到“共享-差异-补偿”三维视角里综合看，是很适合论文展示的一张总结图。

DFlow 的 SHAP 图
DFlow_shap_bar.png
这是 DFlow 的 SHAP 柱状图，显示全局重要性排序。

DFlow_shap_beeswarm.png
这是 DFlow 的 SHAP beeswarm 图。它既看重要性，也看方向性和样本分布。

DFlow_dependence_ASM_A_MeanT.png
DFlow_dependence_ASD_A_SDevT.png
这是 DFlow 下两个最重要特征的 dependence 图。横轴是特征值，纵轴是 SHAP 值。它用来分析特征升高时 DFlow 是被推高还是压低，以及是否存在非线性区间。

DFlow_waterfall_high_prediction.png
DFlow_waterfall_low_prediction.png
这是 DFlow 在测试集中一个高预测样本和一个低预测样本的 waterfall 图。它们把预测值拆成“基线值 + 各特征贡献”，用来解释为什么这个样本预测高、那个样本预测低。

DGap 的 SHAP 图
DGap_shap_bar.png
这是 DGap 的 SHAP 全局柱状图。

DGap_shap_beeswarm.png
这是 DGap 的 SHAP beeswarm 图。

DGap_dependence_CSM_C_MeanT.png
DGap_dependence_ILD_I_SDevT.png
这是 DGap 下最重要两个特征的 dependence 图，一个更偏共享驱动，一个更偏协同/差异驱动。

DGap_waterfall_high_prediction.png
DGap_waterfall_low_prediction.png
这是 DGap 在测试集高预测样本和低预测样本上的局部瀑布图。

局部解释表
dual_output_local_explanations.csv
这是局部解释总表。它把 DFlow 和 DGap 的高预测/低预测样本都列出来，并给出：
输出变量
样本类型（high/low）
测试集内部样本编号
特征名
特征值
SHAP 值
基线值
这张表适合你做精确的单样本机制解释，而且现在已经修正过索引问题，局部解释和 waterfall 图是一致的。

如果一句话概括这些文件的分工：
metric/cv/predictions 看模型效果，
feature_importance/shap_importance 看谁重要，
synergy_indices 看谁是共享驱动、差异驱动或补偿因子，
dependence/beeswarm/waterfall/local_explanations 看方向和单样本机制。
---
# 第二问 双输出 XGBoost + SHAP 协同解释模型结果摘要

- 建模目标：以 16 个脉冲特征为输入，基于 MultiOutputRegressor 封装两个 XGBoost 回归器，对 (DFlow, DGap) 做双输出联合分析，并用 SHAP 分解共享驱动、差异驱动与补偿驱动。
- 最优参数：{'n_estimators': 100, 'max_depth': 3, 'learning_rate': 0.05, 'subsample': 0.8, 'colsample_bytree': 0.8, 'reg_lambda': 3.0, 'gamma': 0.1}

## 测试集性能
- DFlow：RMSE=0.2856，MAE=0.1609，R²=0.4910
- DGap：RMSE=0.3935，MAE=0.3331，R²=0.6441
- 综合平均 RMSE=0.3395，综合平均 R²=0.5676

## 协同解释结果
- 共享驱动因子前 5 名：CSM_C_MeanT, CLM_C_MeanT, ASM_A_MeanT, ASD_A_SDevT, BSM_B_MeanT
- 差异驱动因子前 5 名：CSM_C_MeanT, ILD_I_SDevT, CLM_C_MeanT, BSM_B_MeanT, BSD_B_SDevT
- 补偿型特征（方向协同系数为负且绝对值较大）前 5 名：ASD_A_SDevT, ASM_A_MeanT, BLD_B_SDevT, CSD_C_SDevT, ALD_A_SDevT

## 解释结论
- 该模型在实现上属于“共享输入空间 + 双输出并行回归器”的联合分析框架，而非原生单损失的多目标树模型。
- 若某特征 shared_contribution 高且 differential_contribution 低，则说明它同时影响 DFlow 与 DGap，是协同调控的共享监测指标。
- 若某特征 differential_contribution 高，则说明它更偏向于其中一个控制变量，是差异驱动因子。
- 若某特征 direction_correlation 为负，则说明它对两个输出的 SHAP 方向存在补偿关系，可用于解释主从协同回归中出现的负向联动项。
