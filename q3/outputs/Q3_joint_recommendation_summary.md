
# 文件解释
配置与先验文件
selected_recommendation_config.json
这是第三问最关键的配置文件。它记录了：
最终选中的推荐方法是 full_joint_recommendation 还是 simplified_forward_recommendation
第三问评估阶段和部署阶段分别怎么用双输出模型
DFlow、DGap 的可行离散档位
可行边界
协同先验参数
最终选中的 lambda1、lambda2、lambda3
这张文件最适合用来回答“第三问到底是按什么规则推荐参数的”。

collaboration_prior.csv
这是协同先验表。它来自第二问主从协同回归和第二问 SHAP 协同指标，里面有：
Intercept
beta
协同中心项里每个特征的原始系数
每个特征对应的 synergy_weight
以及二者相乘后的 effective_coefficient
它说明第三问里“协同中心项”到底是怎么构造出来的。

recommendation_equations.txt
这是第三问优化模型的文字版公式说明。包括：
目标函数形式
最终采用的方法
选中的权重参数
协同先验方程
允许的档位和可行边界
适合直接拿去写论文公式说明部分。

搜索与指标结果文件
lambda_search_full.csv
这是“完整联合推荐模型”的权重搜索结果。每一行是一组 lambda1, lambda2, lambda3，并记录该组参数下的：
rmse_dflow
rmse_dgap
joint_rmse_mean
joint_pair_accuracy
collaboration_rmse
它用来比较不同权重下第三问的推荐效果。

lambda_search_simplified.csv
这和上面类似，但对应的是“简化主从推荐模型”，即只固定 F* = \hat F、重点修正 G* 的那种形式。

recommendation_metrics_summary.csv
这是第三问最核心的总指标表。它汇总了三种方法在训练集和测试集上的表现：
base_joint_prediction
full_joint_recommendation
simplified_forward_recommendation
并且每种方法又区分：
continuous：连续推荐值效果
discrete：投影到离散档位后的效果
主要字段包括：
rmse_dflow
rmse_dgap
joint_rmse_mean
joint_pair_accuracy
collaboration_rmse
这张表最适合回答“第三问推荐模型和纯预测相比有什么区别”。

推荐结果文件
recommendations_test.csv
这是测试集样本的推荐结果表。每一行对应一个测试样本，包含：
真实 DFlow、DGap
基础预测值
最终连续推荐值
最终离散推荐值
推荐值相对基础预测值的调整量
协同残差
它适合做“第三问推荐在测试集上的效果”分析。

recommendations_all_samples.csv
这是对全体样本生成的最终推荐结果表。它不是只看测试集，而是基于第三问最终部署链路，对所有样本都输出一组推荐设定值。适合用来做全局推荐分析或后续统计。

representative_recommendations.csv
这是代表性样本推荐表。它专门挑出“调整幅度最大”的样本，展示：
原始真实值
基础预测值
最终推荐值
调整量
它适合论文里专门举几个样本解释“第三问为什么会修正”。

图表文件
method_metric_comparison.png
这是三种方法的指标对比图。能直观看出：
纯联合预测
完整联合推荐
简化主从推荐
在 DFlow、DGap 和联合指标上的差别。

adjustment_distribution.png
这是推荐调整量分布图。它展示第三问相对于基础预测值，通常会把 DFlow 和 DGap 往哪个方向调、调多少。

base_vs_final_recommendation.png
这是基础预测值和最终推荐值的散点对比图。点越偏离对角线，说明第三问协同校正越明显。它适合说明“第三问不是简单复述第二问预测值，而是在做修正”。

DFlow_level_transition_heatmap.png
这是 DFlow 从“基础离散预测档位”到“最终推荐档位”的转移热力图。它用来观察第三问是否经常把 DFlow 从某一档修正到另一档。

DGap_level_transition_heatmap.png
这和上面类似，但对应 DGap。由于第三问的协同核心主要落在 DGap 修正上，这张图通常更重要。

top_adjustment_samples.png
这是调整幅度最大的样本柱状图。它能直观看出哪些样本被第三问修正得最多，以及修正主要落在 DFlow 还是 DGap 上。

摘要文件
Q3_joint_recommendation_summary.md
这是第三问的文字版总结。它会概括：
第三问模型定位
协同先验来源
第二问 SHAP 协同权重如何进入第三问
最终选择了哪种推荐模型
最优权重是什么
测试集上的关键结果
以及第三问最适合如何在论文中表述

如果一句话概括这些文件：
config/prior/equations 说明第三问模型怎么建，
lambda_search/metrics 说明第三问怎么选权重、效果如何，
recommendations 给出具体推荐值，
各类 png 图展示“推荐值是怎么从基础预测修正过来的”，
summary.md 则是整套第三问的结论摘要。
---
# 第三问 联合预测—协同校正参数推荐模型结果摘要

- 模型定位：沿用第二问双输出 XGBoost 的最优结构与参数配置，并在第三问当前划分下重新训练双输出模型以获得基础联合预测值；再利用第二问正向主从协同回归构造 DGap 协同中心项，通过二次目标函数把“预测值”修正为“推荐设定值”。
- 协同先验方程：G ≈ c(x) + βF，其中 β = -0.5026。
- 协同中心项特征：CSM_C_MeanT, CLM_C_MeanT, BLD_B_SDevT, CSD_C_SDevT, C；其中 C 对应 ILD_I_SDevT。
- SHAP 协同权重：{'CSM_C_MeanT': 0.17342662468222342, 'CLM_C_MeanT': 0.32819447632959753, 'C': 0.07721083811225438, 'CSD_C_SDevT': 0.17554948609588833, 'BLD_B_SDevT': 0.24561857478003632}。
- 双输出基础预测训练策略：loaded_existing_q2_model，封装器为 MultiOutputRegressor，基学习器为 XGBRegressor。
- 第一问 DFlow GRA 前五：ASM_A_MeanT, ALM_A_MeanT, ALD_A_SDevT, CLD_C_SDevT, ILD_I_SDevT。
- 第一问 DGap GRA 前五：CSM_C_MeanT, CLM_C_MeanT, BLD_B_SDevT, ILD_I_SDevT, CSD_C_SDevT。
- 第二问共享/差异/补偿分析中最值得关注的协同特征前五：CSM_C_MeanT, CLM_C_MeanT, ASM_A_MeanT, ASD_A_SDevT, BSM_B_MeanT。

## 训练集上最优权重搜索结果
- Full joint 推荐最优：lambda1=1.0, lambda2=4.0, lambda3=0.25，离散 joint_rmse_mean=0.1418，joint_pair_accuracy=0.9593。
- Simplified forward 推荐最优：lambda2=4.0, lambda3=0.25，离散 joint_rmse_mean=0.1562，joint_pair_accuracy=0.9512。

## 测试集回顾性评价
- 最终选择的方法：full_joint_recommendation。
- 选择依据：在离散设定值评价下具有更低的 joint_rmse_mean 或更高的 joint_pair_accuracy，同时保持较低的协同残差。
- 当前最佳测试集离散结果：DFlow RMSE=0.3592，DGap RMSE=0.4399，joint_pair_accuracy=0.7097，collaboration_rmse=0.9500。

## 第三问的建模结论
- 第三问不是重新做特征筛选，而是把第二问的基础联合预测值转化为最终可执行设定值。
- 在当前数据中，推荐值必须进一步投影到历史可行档位 {-1, 0, 1}，这样才能与真实控制设定保持一致。
- 协同校正后，推荐值相较于纯联合预测会主动压缩违反主从联动关系的组合，从而降低 DGap 对 DFlow 负向协同关系的偏离。
- 若论文更强调工程可执行性，可优先使用 simplified_forward_recommendation；若更强调双变量同时折中，可展示 full_joint_recommendation 作为对照。

## 代表性样本推荐结果
```text
 sample_index  DFlow_actual  DGap_actual  DFlow_base_pred  DGap_base_pred  DFlow_recommended_discrete  DGap_recommended_discrete  delta_DFlow_continuous  delta_DGap_continuous  joint_adjustment_abs
           77           0.0         -1.0        -0.006738       -0.901006                         0.0                       -1.0                0.183086               0.091062              0.274148
           14           0.0         -1.0        -0.020638       -0.887753                         0.0                       -1.0                0.182973               0.091005              0.273978
           40           0.0         -1.0         0.000386       -0.900012                         0.0                       -1.0                0.182598               0.090819              0.273417
           16           0.0         -1.0        -0.004467       -0.890783                         0.0                       -1.0                0.181920               0.090482              0.272402
           39           0.0         -1.0        -0.130618       -0.817120                         0.0                       -1.0                0.181586               0.090315              0.271901
           56           0.0         -1.0        -0.006147       -0.881184                         0.0                       -1.0                0.180863               0.089956              0.270819
           60           0.0         -1.0        -0.006740       -0.861011                         0.0                       -1.0                0.179137               0.089098              0.268235
           51           0.0         -1.0         0.000216       -0.864198                         0.0                       -1.0                0.179102               0.089080              0.268181
          138           0.0         -1.0        -0.013888       -0.848018                         0.0                       -1.0                0.178234               0.088648              0.266882
           55           0.0         -1.0         0.018420       -0.869835                         0.0                       -1.0                0.178227               0.088645              0.266871
           15           0.0         -1.0        -0.017795       -0.838716                         0.0                       -1.0                0.177343               0.088205              0.265548
           76           0.0         -1.0         0.018555       -0.857562                         0.0                       -1.0                0.177212               0.088140              0.265353
```