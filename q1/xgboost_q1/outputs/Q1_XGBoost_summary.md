# 结果日志
## 日志1
这次实现的是你要求的双模型框架：
DFlow 基线模型：16 个特征
DFlow 优化模型：GRA 前 5 特征
DGap 基线模型：16 个特征
DGap 优化模型：GRA 前 5 特征

核心结果文件有：
model_metrics_summary.csv
best_params.json
feature_sets.csv
Q1_XGBoost_summary.md

预测与图表也都生成了，包括每个模型的：
实际值 vs 预测值散点图
特征重要性条形图
测试集指标对比图
模型预测结果 CSV
模型文件 JSON

测试集上的关键指标是：
DFlow 基线：RMSE 0.2483，MAE 0.1373，R² 0.5941
DFlow 优化：RMSE 0.2452，MAE 0.1332，R² 0.6044
DGap 基线：RMSE 0.3628，MAE 0.2819，R² 0.7086
DGap 优化：RMSE 0.3709，MAE 0.3115，R² 0.6954

结论很清楚：DFlow 上，GRA 筛选后的优化模型略优于全特征基线模型；DGap 上，则是全特征基线模型更好。这说明 GRA 对 DFlow 的降维更有效，而 DGap 可能还保留了一部分前五特征之外的信息。

几个比较适合论文展示的图是：
DFlow_metric_comparison.png
DGap_metric_comparison.png
DFlow_optimized_feature_importance.png
DGap_baseline_feature_importance.png

## 日志2
现在 xgboost_q1.py 里已经改成了标准回归写法：
去掉了 train_test_split(..., stratify=y)
去掉了 StratifiedKFold
改为 KFold(n_splits=5, shuffle=True, random_state=42)

我也已经重新跑完了全部 4 个模型，所以这些文件都已经同步更新：
model_metrics_summary.csv
Q1_XGBoost_summary.md
以及所有预测图、特征重要性图、模型文件和预测结果表。

更新后的测试集指标是：
DFlow 基线：RMSE 0.2967，MAE 0.1577，R² 0.4507
DFlow 优化：RMSE 0.2916，MAE 0.1859，R² 0.4693
DGap 基线：RMSE 0.3835，MAE 0.3348，R² 0.6619
DGap 优化：RMSE 0.4164，MAE 0.3547，R² 0.6014

所以修正后结论仍然是：
DFlow 上优化模型略优，
DGap 上基线模型更优。

---
# 文件解释
feature_sets.csv
这是建模用特征清单。里面记录了每个目标变量在“基线模型”和“优化模型”下分别用了哪些特征。
baseline 用 16 个全部特征。
optimized 用 GRA 前 5 特征。

model_metrics_summary.csv
这是最重要的总指标表。它汇总了 4 个模型在训练集和测试集上的 RMSE、MAE、R²、特征数等信息。你要比较“基线模型”和“优化模型”谁更好，主要看这张表。

best_params.json
这是调参结果文件。里面记录了每个模型最优的 n_estimators、max_depth、learning_rate、subsample 等参数，以及交叉验证下的最优 RMSE。适合在论文里说明“模型经过参数搜索优化”。

Q1_XGBoost_summary.md
这是文字版摘要，概括了建模策略、测试集表现和各模型的最优参数。适合快速看结论。

预测结果文件
DFlow_baseline_predictions.csv
DFlow_optimized_predictions.csv
DGap_baseline_predictions.csv
DGap_optimized_predictions.csv
这 4 个文件分别对应 4 个模型的样本级预测结果。字段通常有：
dataset：训练集或测试集
actual：真实值
predicted：预测值
residual：残差
它们主要用于误差分析、画散点图，以及检查模型预测偏差。

模型文件
DFlow_baseline_model.json
DFlow_optimized_model.json
DGap_baseline_model.json
DGap_optimized_model.json
这 4 个文件是训练好的 XGBoost 模型本体，可以以后直接加载复用，不需要重新训练。

预测效果图
DFlow_baseline_actual_vs_pred.png
DFlow_optimized_actual_vs_pred.png
DGap_baseline_actual_vs_pred.png
DGap_optimized_actual_vs_pred.png
这 4 张图是“真实值 vs 预测值”散点图。点越接近对角线，说明预测越准确。它们主要用来直观看模型拟合效果。

特征重要性结果
DFlow_baseline_feature_importance.csv
DFlow_optimized_feature_importance.csv
DGap_baseline_feature_importance.csv
DGap_optimized_feature_importance.csv
这是 4 个模型的特征重要性数值表，记录每个输入变量在对应模型里的贡献大小。

DFlow_baseline_feature_importance.png
DFlow_optimized_feature_importance.png
DGap_baseline_feature_importance.png
DGap_optimized_feature_importance.png
这 4 张图是对应的特征重要性条形图，方便直观看“谁更重要”。

模型对比图
DFlow_metric_comparison.png
DGap_metric_comparison.png
这两张图专门用来比较每个目标变量下“基线模型”和“优化模型”的测试集表现。图里一般同时展示 RMSE、MAE、R²，是最适合放在论文对比部分的图。
---
# 第一问 XGBoost 建模结果摘要

- 建模策略：采用“16个特征的基线模型 + GRA前5特征的优化模型”双模型框架。
- 数据划分：按照 80% 训练集、20% 测试集进行随机划分。
- 参数搜索：采用 RandomizedSearchCV，并使用 5 折 KFold 交叉验证。

## 测试集表现
- DFlow 基线模型：RMSE=0.2967，MAE=0.1577，R²=0.4507
- DFlow 优化模型：RMSE=0.2916，MAE=0.1859，R²=0.4693
- 按 RMSE 指标判断，DFlow 的较优模型为：优化模型。

- DGap 基线模型：RMSE=0.3835，MAE=0.3348，R²=0.6619
- DGap 优化模型：RMSE=0.4164，MAE=0.3547，R²=0.6014
- 按 RMSE 指标判断，DGap 的较优模型为：基线模型。

## 最优参数
- DFlow 基线模型：CV_RMSE=0.2775，交叉验证折数=5，最优参数={'subsample': 0.7, 'reg_lambda': 0.5, 'n_estimators': 100, 'min_child_weight': 1, 'max_depth': 5, 'learning_rate': 0.03, 'gamma': 0.05, 'colsample_bytree': 0.85}
- DFlow 优化模型：CV_RMSE=0.2839，交叉验证折数=5，最优参数={'subsample': 0.7, 'reg_lambda': 1.0, 'n_estimators': 60, 'min_child_weight': 4, 'max_depth': 3, 'learning_rate': 0.12, 'gamma': 0.1, 'colsample_bytree': 0.85}
- DGap 基线模型：CV_RMSE=0.4729，交叉验证折数=5，最优参数={'subsample': 1.0, 'reg_lambda': 5.0, 'n_estimators': 100, 'min_child_weight': 1, 'max_depth': 2, 'learning_rate': 0.05, 'gamma': 0.1, 'colsample_bytree': 0.85}
- DGap 优化模型：CV_RMSE=0.4900，交叉验证折数=5，最优参数={'subsample': 1.0, 'reg_lambda': 0.5, 'n_estimators': 100, 'min_child_weight': 2, 'max_depth': 3, 'learning_rate': 0.05, 'gamma': 0.1, 'colsample_bytree': 0.7}
