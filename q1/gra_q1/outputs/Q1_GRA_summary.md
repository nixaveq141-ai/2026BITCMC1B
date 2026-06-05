# 结果日志

已按你的要求在 q1 中完成第 1 问灰色关联分析的代码、结果表和可视化图表生成。主脚本是 gra_q1.py，它会读取 预处理后数据_未标准化.csv，再按你给出的模型在脚本内部做极差标准化、GRA 计算和 Spearman 方向增强。

生成的核心结果文件包括：
DFlow_gra_results.csv
DGap_gra_results.csv
Q1_gra_combined_results.csv
Q1_GRA_summary.md

图表文件包括：
DFlow_gra_ranking.png
DGap_gra_ranking.png
DFlow_signed_influence.png
DGap_signed_influence.png
DFlow_coefficient_heatmap.png
DGap_coefficient_heatmap.png
DFlow_DGap_gra_comparison.png

当前算出的关键结果是：
DFlow 前 5 位特征为 ASM_A_MeanT、ALM_A_MeanT、ALD_A_SDevT、CLD_C_SDevT、ILD_I_SDevT。
DGap 前 5 位特征为 CSM_C_MeanT、CLM_C_MeanT、BLD_B_SDevT、ILD_I_SDevT、CSD_C_SDevT。
两个目标变量共同进入前 5 的特征是 ILD_I_SDevT，说明它可能是后续协同调控里值得重点关注的共同关键因子。

如果你愿意，我下一步可以继续基于这些 GRA 结果，直接帮你写“第一问”的论文正文分析与结论段。

# 图表说明


[DFlow_gra_ranking.png](C:\Users\33056\PycharmProjects\2026BITCMC1B\q1\DFlow_gra_ranking.png)
这是一张 `DFlow` 的灰色关联度排序条形图。横轴是灰色关联度，纵轴是 16 个脉冲特征。条越长，说明该特征与 `DFlow` 的变化轨迹越接近、关联越强。它主要用来回答“哪些特征对 DFlow 影响更大”。

[DGap_gra_ranking.png](C:\Users\33056\PycharmProjects\2026BITCMC1B\q1\DGap_gra_ranking.png)
这和上面同理，只不过目标变量换成了 `DGap`。它用来识别“哪些脉冲特征对放电间隙 DGap 更敏感、更重要”。

[DFlow_signed_influence.png](C:\Users\33056\PycharmProjects\2026BITCMC1B\q1\DFlow_signed_influence.png)
这是 `DFlow` 的带符号影响指数图。它把“灰色关联度大小”和“Spearman 相关系数正负号”结合起来了。横轴为带符号影响指数，右侧为正值，表示该特征与 `DFlow` 总体呈正向关系；左侧为负值，表示总体呈负向关系。绝对值越大，说明影响越强。

[DGap_signed_influence.png](C:\Users\33056\PycharmProjects\2026BITCMC1B\q1\DGap_signed_influence.png)
这张图和上一张的逻辑完全一样，只是针对 `DGap`。它不仅告诉你“谁重要”，还告诉你“是正向作用还是负向作用”。

[DFlow_coefficient_heatmap.png](C:\Users\33056\PycharmProjects\2026BITCMC1B\q1\DFlow_coefficient_heatmap.png)
这是 `DFlow` 的灰色关联系数热力图。横轴是样本编号，纵轴是 16 个脉冲特征，颜色深浅表示该样本点上该特征与 `DFlow` 的局部相似程度。颜色越深，说明在该样本点上它们越接近。它适合看“某个特征是不是在大多数样本上都稳定相关”。

[DGap_coefficient_heatmap.png](C:\Users\33056\PycharmProjects\2026BITCMC1B\q1\DGap_coefficient_heatmap.png)
这张图对应 `DGap` 的灰色关联系数热力图，用法和上一张一样，只是参考序列换成了 `DGap`。

[DFlow_DGap_gra_comparison.png](C:\Users\33056\PycharmProjects\2026BITCMC1B\q1\DFlow_DGap_gra_comparison.png)
这是一张对比图，把每个特征对 `DFlow` 和对 `DGap` 的灰色关联度并排展示。它最适合回答“同一个特征到底更影响 DFlow 还是更影响 DGap”，也方便找出两个控制变量的共同关键因子和差异因子。

DFlow_coefficient_heatmap.png
这是一张针对 DFlow 的“聚合版灰色关联系数热力图”。纵轴是 16 个脉冲特征，横轴是 5 个聚合统计量：mean、median、std、min、max。颜色越深表示该统计值越大。它的用途是从整体上判断每个特征与 DFlow 的关联水平和稳定性，而不是看每个样本点的细节。

DGap_coefficient_heatmap.png
这张图和上一张逻辑完全相同，只不过目标变量换成了 DGap。它用来观察每个脉冲特征与 DGap 的整体关联强弱、波动情况和极值范围。

[gra_minmax_standardized_data.csv](C:\Users\33056\PycharmProjects\2026BITCMC1B\q1\gra_minmax_standardized_data.csv)
这是做 GRA 时实际使用的标准化数据表。这里采用的是极差标准化，把 16 个脉冲特征以及 `DFlow`、`DGap` 都映射到 `[0,1]` 区间。它的作用是消除量纲影响，让灰色关联分析只比较“变化趋势是否接近”。

[DFlow_gra_results.csv](C:\Users\33056\PycharmProjects\2026BITCMC1B\q1\DFlow_gra_results.csv)
这是以 `DFlow` 为目标变量的最终结果表。每一行对应一个脉冲特征，主要字段含义是：
`feature`：特征名；
`gra_degree`：该特征与 `DFlow` 的灰色关联度，越大表示关联越强；
`spearman_rho`：Spearman 相关系数，用来判断正负方向；
`signed_influence`：带符号影响指数，等于关联度乘以方向符号；
`rank`：按关联度从大到小的排名；
`relationship_direction`：正向或负向；
`association_level`：关联强弱分级。
这张表是回答“谁对 DFlow 更重要”的核心表。

[DGap_gra_results.csv](C:\Users\33056\PycharmProjects\2026BITCMC1B\q1\DGap_gra_results.csv)
这张表和上面完全同理，只是目标变量换成了 `DGap`。它用来回答“谁对 DGap 更重要”。

[Q1_gra_combined_results.csv](C:\Users\33056\PycharmProjects\2026BITCMC1B\q1\Q1_gra_combined_results.csv)
这是把 `DFlow` 和 `DGap` 两套结果合并后的总表。它方便你统一筛选、排序和横向比较两个目标变量下的特征表现，特别适合后面写“共同关键因子”和“差异性因子”的分析。

[DFlow_gra_coefficients_by_sample.csv](C:\Users\33056\PycharmProjects\2026BITCMC1B\q1\DFlow_gra_coefficients_by_sample.csv)
这是 `DFlow` 的样本级灰色关联系数表。每一行对应一个样本，每一列对应一个特征，单元格的值表示“该特征在该样本点上与 DFlow 的局部关联程度”。它比最终关联度更细，可以看出某个特征是不是只在部分样本上作用明显。

[DGap_gra_coefficients_by_sample.csv](C:\Users\33056\PycharmProjects\2026BITCMC1B\q1\DGap_gra_coefficients_by_sample.csv)
这张表是 `DGap` 的样本级灰色关联系数表，含义与上表一致，只是参考对象变成了 `DGap`。

[Q1_GRA_summary.md](C:\Users\33056\PycharmProjects\2026BITCMC1B\q1\Q1_GRA_summary.md)
这是一份文字版摘要，提炼了 `DFlow` 和 `DGap` 的前五关键特征、方向信息和整体解读。它更像一个快速结论文件，方便直接查看主要结果。



# 问题一灰色关联分析结果

- 数据来源：预处理后数据_未标准化.csv
- 方法：极差标准化 + 灰色关联分析（GRA） + Spearman方向增强
- 分辨系数：0.5
- 输入特征：16个脉冲特征参数
- 目标变量：DFlow、DGap

## DFlow 前五特征
- ASM_A_MeanT：灰色关联度=0.7354，Spearman=-0.0618，带符号影响指数=-0.7354，方向=Negative
- ALM_A_MeanT：灰色关联度=0.7004，Spearman=-0.2294，带符号影响指数=-0.7004，方向=Negative
- ALD_A_SDevT：灰色关联度=0.6955，Spearman=0.2057，带符号影响指数=0.6955，方向=Positive
- CLD_C_SDevT：灰色关联度=0.6817，Spearman=0.2865，带符号影响指数=0.6817，方向=Positive
- ILD_I_SDevT：灰色关联度=0.6811，Spearman=0.1969，带符号影响指数=0.6811，方向=Positive

## DGap 前五特征
- CSM_C_MeanT：灰色关联度=0.7291，Spearman=0.6796，带符号影响指数=0.7291，方向=Positive
- CLM_C_MeanT：灰色关联度=0.7169，Spearman=0.6190，带符号影响指数=0.7169，方向=Positive
- BLD_B_SDevT：灰色关联度=0.6918，Spearman=0.4320，带符号影响指数=0.6918，方向=Positive
- ILD_I_SDevT：灰色关联度=0.6888，Spearman=0.4377，带符号影响指数=0.6888，方向=Positive
- CSD_C_SDevT：灰色关联度=0.6827，Spearman=0.4726，带符号影响指数=0.6827，方向=Positive

## 结果解读
- 两个目标变量前五集合的共同关键特征为：ILD_I_SDevT。
- 带符号影响指数大于0表示该特征与目标变量整体呈正向关系，小于0表示整体呈负向关系。
- 灰色关联度刻画的是变化趋势接近程度，不代表严格因果；Spearman符号用于补充方向解释。
