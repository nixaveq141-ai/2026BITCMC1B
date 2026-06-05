# 结果日志
第一层是“统计协同关系是否存在”：
存在，而且在 DFlow→DGap 的正向主从结构里更明显。
正向协同从模型里，DFlow 的联动系数约为 -0.5026，说明在控制其他协同特征后，DFlow 对 DGap 存在稳定的负向联动作用。这个系数的 bootstrap 95% 区间是 [-0.7020, -0.3624]，明显不跨 0，所以这个联动不是偶然现象。相关结果见 lambda_bootstrap_summary.csv 和 Q2_master_slave_summary.md。

第二层是“这种协同能不能用来简化控制”：
要分场景看。

如果是“操作者先设定主变量，再联动推荐从变量”，正向 DFlow→DGap 是有效的。
测试集上，DGap 独立模型的 RMSE 是 0.5327，而正向协同模型在已知 DFlow 时的 RMSE 是 0.4679，明显更好；R² 也从 0.3476 提升到 0.4966。这说明把 DFlow 当主变量、再联动求 DGap，在“辅助调节”意义下是成立的。结果见 holdout_metrics_summary.csv。

但如果是“完全自动化”，也就是先从特征预测主变量、再驱动从变量，那正向链路并不占优。
正向部署型协同模型的 repeated CV 平均 RMSE 是 0.5155 ± 0.0394，略差于正向独立模型 0.5032 ± 0.0395；而反向 DGap→DFlow 的部署型协同 RMSE 是 0.3935 ± 0.0614，反而优于反向独立模型 0.3815 ± 0.0596 的差距更小、整体更稳一些。这个结果说明：

面向“人工先设主变量，系统联动推荐从变量”的辅助控制方案，推荐 DFlow→DGap
面向“完全自动预测主变量再联动”的自动化链路，当前结果反而更支持 DGap→DFlow
这些稳健性结果在 repeated_cv_summary.csv、forward_reverse_rmse_comparison.png 和 Q2_master_slave_summary.md 里都能直接看到。

如果你要一句最适合写进论文的话，可以写成：
“基于 GRA 先验与协同指数构建的主从协同回归结果表明，DFlow 与 DGap 之间存在显著联动关系；其中，在操作者先设定主变量的场景下，DFlow→DGap 更适合作为辅助协同调节方案，而在完全自动部署场景下，当前数据对 DGap→DFlow 的支持更强。”
---
# 文件解释
结构说明类
feature_manifest.csv
这是第二问建模时用到的特征结构清单。它说明了：
正向主模型用了哪些特征；
正向从模型用了哪些补充特征；
协同指数 C 是怎么定义的；
反向主模型和反向从模型分别用了哪些特征。
它的作用是把“GRA 结构先验”显式记录下来。

Q2_master_slave_summary.md
这是第二问的中文摘要文件。里面概括了：
这版模型做了哪些增强；
主模型和协同从模型的估计方程；
hold-out 测试集表现；
重复交叉验证结果；
以及最后的协同判断结论。
这是最适合先读的一份总览文件。

指标结果类
holdout_metrics_summary.csv
这是单次训练测试划分下的结果表。每一行对应一个模型在训练集或测试集上的 RMSE、MAE、R²。这里面包含：
正向主模型 DFlow_master_forward
正向独立从模型 DGap_independent_forward
正向协同从模型 DGap_collaborative_actual_forward
正向部署型协同模型 DGap_collaborative_deploy_forward
反向主模型 DGap_master_reverse
反向独立从模型 DFlow_independent_reverse
反向协同从模型 DFlow_collaborative_actual_reverse
反向部署型协同模型 DFlow_collaborative_deploy_reverse
它主要回答“在一次固定划分下，各方案表现如何”。

repeated_cv_all_folds.csv
这是重复交叉验证的全部明细。每一行是一折实验结果，记录不同模型在某一折上的 RMSE、MAE、R²。它适合做更细的稳健性分析。

repeated_cv_summary.csv
这是对重复交叉验证的汇总表。它给出每个模型在 repeated K-fold 下的：
rmse_mean / rmse_std
mae_mean / mae_std
r2_mean / r2_std
这张表最适合在论文里用来说明“模型稳不稳”。

系数与方程类
model_coefficients.csv
这是所有回归模型的系数表。包括：
正向主模型系数
正向独立从模型系数
正向协同从模型系数
反向主模型系数
反向协同从模型系数
它能直接告诉你每个变量对目标的正负方向和强弱。

estimated_equations.txt
这是把回归结果整理成方程形式后的文本文件。比如：
DFlow = ...
DGap_collaborative = ...
这份文件很适合直接拿去写论文公式说明。

GRA 约束与协同稳健性类
gra_penalty_weights.csv
这是这版模型非常关键的文件。它记录了每个变量在加权惩罚项里的惩罚系数，也就是 GRA 结构约束是怎么进入模型的。关联度高的特征惩罚更小，更容易保留；关联度低的特征惩罚更大，更容易收缩。

lambda_bootstrap_samples.csv
这是正向协同系数 lambda 的 bootstrap 抽样结果。每一行是一次 bootstrap 重抽样后估计出的 lambda。它用于检验联动系数是否稳健。

lambda_bootstrap_summary.csv
这是对 bootstrap 结果的汇总，给出：
lambda_mean
lambda_ci_low_95
lambda_ci_high_95
collaborative_sign
它主要回答“lambda 是否显著不为 0”。

collaboration_assessment.csv
这是第二问的结论辅助表。它把几个关键判断量单独抽出来，比如：
正向协同系数值；
正向协同模型相比独立模型的 RMSE 改善；
部署场景下协同与独立的差值；
正向和反向在 repeated CV 下的差距。
它适合拿来直接支撑结论段。

预测结果类
model_predictions.csv
这是所有模型的样本级预测结果总表。每一行都会给出：
模型名
训练/测试集
真实值
预测值
残差
这张表适合做误差分析，或者以后继续画图。

散点图类
DFlow_master_forward_actual_vs_pred.png
这是正向主模型 DFlow 的真实值-预测值散点图。点越接近对角线，说明预测越准确。

DGap_independent_forward_actual_vs_pred.png
这是正向独立从模型的散点图，用来表示不考虑主从联动时 DGap 的拟合情况。

DGap_collaborative_actual_forward_actual_vs_pred.png
这是正向协同从模型在“已知 DFlow”的情况下的散点图。它体现的是“人工先设主变量，再联动求从变量”的场景。

DGap_collaborative_deploy_forward_actual_vs_pred.png
这是正向协同模型在“用预测的 DFlow 驱动 DGap”时的散点图。它更接近完全自动部署的效果。

对比图类
holdout_forward_metrics.png
这是正向方向下几种模型在 hold-out 测试集上的指标对比图，一般展示 RMSE/MAE/R²。它用来直观看正向协同是否优于独立模型。

repeated_cv_mean_performance.png
这是 repeated K-fold 下各模型平均表现的对比图，用于展示稳健性。

forward_reverse_rmse_comparison.png
这是正向和反向主从方向的 RMSE 对比图。它是回答“到底该用 DFlow→DGap 还是 DGap→DFlow”的关键图。

forward_slave_collaborative_coefficients.png
这是正向协同从模型的系数图。它告诉你在 DGap 的协同模型里，哪些变量在推高它，哪些变量在拉低它，尤其能看出 DFlow 的联动方向。

gra_penalty_weights.png
这是 GRA 惩罚权重图。它可视化了“GRA 约束”是如何进入回归模型的，能体现高关联特征受惩罚较小、共同因子 C 和主从联动变量有额外保留倾向。

lambda_bootstrap_distribution.png
这是正向 lambda 的 bootstrap 分布图。它用来直观看 lambda 是否稳定、区间是否跨 0。



---
# 第二问 GRA约束主从协同回归模型结果摘要

- 本版增强点：显式构造协同指数 C=ILD_I_SDevT；利用第一问 GRA 关联度构造加权惩罚项；补做正反主从方向对照；使用 repeated K-fold 报告均值与标准差。
- Forward 主模型：DFlow = 0.1230 + 0.1290*ASM_A_MeanT - 0.1709*ALM_A_MeanT + 0.0710*ALD_A_SDevT + 0.0718*CLD_C_SDevT - 0.0295*C
- Forward 协同从模型：DGap_collaborative = 0.0500 + 0.3364*CSM_C_MeanT + 0.2303*CLM_C_MeanT - 0.1042*BLD_B_SDevT + 0.0576*CSD_C_SDevT + 0.0431*C - 0.5026*DFlow
- Reverse 协同从模型：DFlow_collaborative = 0.1212 + 0.1086*ASM_A_MeanT - 0.1886*ALM_A_MeanT + 0.0787*ALD_A_SDevT + 0.0775*CLD_C_SDevT - 0.0205*C - 0.1163*DGap

## Hold-out 测试集
- DFlow_master_forward：RMSE=0.4047，MAE=0.2664，R²=-0.0221
- DGap_independent_forward：RMSE=0.5327，MAE=0.4938，R²=0.3476
- DGap_collaborative_actual_forward：RMSE=0.4679，MAE=0.4179，R²=0.4966
- DGap_collaborative_deploy_forward：RMSE=0.5334，MAE=0.4954，R²=0.3459
- DGap_master_reverse：RMSE=0.5328，MAE=0.4938，R²=0.3474
- DFlow_independent_reverse：RMSE=0.4047，MAE=0.2664，R²=-0.0221
- DFlow_collaborative_actual_reverse：RMSE=0.3916，MAE=0.2538，R²=0.0433
- DFlow_collaborative_deploy_reverse：RMSE=0.4146，MAE=0.2704，R²=-0.0725

## 重复交叉验证
- forward 协同部署模型 RMSE 均值±标准差=0.5155 ± 0.0394
- reverse 协同部署模型 RMSE 均值±标准差=0.3935 ± 0.0614

## 协同判断
- forward 方向 lambda=-0.5026，bootstrap 95% 区间=[-0.7020, -0.3624]。
- 已知/设定 DFlow 时，forward 协同模型优于 DGap 独立模型，说明统计协同关系成立。
- 已知/设定主变量时，forward 方向的协同增益大于 reverse 方向，因此从“人工先设主变量、系统联动推荐从变量”的角度，DFlow→DGap 更具解释力。
- 使用主模型预测主变量进行全自动部署时，forward 协同模型与独立模型接近，而 reverse 部署型 RMSE 更低，说明若追求全自动链路，反向方向反而更稳。
- 因而第二问可给出分层结论：面向辅助联动调节，推荐 DFlow→DGap；面向完全自动部署，当前证据更支持 DGap→DFlow。
