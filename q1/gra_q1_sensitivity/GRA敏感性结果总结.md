# 结果日志
这次实验的结论很清楚：
DFlow 的前五特征在 rho=0.3/0.5/0.7 下完全一致，稳定性非常强：
ASM_A_MeanT、ALM_A_MeanT、ALD_A_SDevT、CLD_C_SDevT、ILD_I_SDevT

DGap 的前五特征总体也比较稳定，其中有 4 个特征在三个 rho 下都进入前五：
CSM_C_MeanT、CLM_C_MeanT、BLD_B_SDevT、ILD_I_SDevT
第 5 个位置在 CSD_C_SDevT 和 CLD_C_SDevT 之间发生了小幅变化。

所以从敏感性角度看：
DFlow 的 GRA 排序结论非常稳健。
DGap 的核心主导特征也较稳健，但边界特征存在轻微排序波动。
---
# 文件解释
结果表
gra_rho_sensitivity_full_results.csv
这是最完整的敏感性实验明细表。它记录了在 rho=0.3、0.5、0.7 下，每个特征相对于 DFlow 和 DGap 的灰色关联度 gra_degree 和对应排名 rank。如果你想精确查“某个特征在不同 rho 下的名次有没有变化”，看这张表。

gra_rho_top5_summary.csv
这是前五特征汇总表。每一行对应一个目标变量和一个 rho 值，直接列出该 rho 下的前五特征。它主要回答“换了分辨系数之后，前五名单有没有变”。

gra_rho_top5_stability_summary.csv
这是稳定性统计表。它把每个特征在三个 rho 下的排名和是否进入前五做了汇总。
里面关键字段可以这样看：
rank_0.3 / rank_0.5 / rank_0.7：该特征在各 rho 下的排名；
rho_0.3 / rho_0.5 / rho_0.7：是否进入前五，进入记为 1，否则为 0；
top5_count：该特征在三个 rho 中有几次进入前五；
mean_rank：三个 rho 下平均排名。
这张表最适合用来判断“哪些特征是稳定关键特征”。

图表
DFlow_rho_degree_sensitivity.png
这是 DFlow 的“关联度敏感性热力图”。纵轴是特征，横轴是 rho=0.3、0.5、0.7，颜色和数字表示灰色关联度大小。它用来观察“改 rho 后，每个特征的关联度数值变化是否明显”。

DGap_rho_degree_sensitivity.png
这张图和上面相同，只是对象换成了 DGap。

DFlow_rho_rank_sensitivity.png
这是 DFlow 的“排名敏感性热力图”。它展示每个特征在不同 rho 下的排名变化。适合判断“排序结果是否稳”。

DGap_rho_rank_sensitivity.png
这是 DGap 的对应排名敏感性热力图。

DFlow_top5_stability.png
这张图专门看 DFlow 前五特征的稳定性。横轴是 rho_0.3、rho_0.5、rho_0.7，数值为 1 表示该特征在对应 rho 下进入前五，0 表示未进入。颜色越集中为 1，说明越稳定。

DGap_top5_stability.png
这张图是 DGap 的前五稳定性图，用法和上面一样。

说明文档
GRA_sensitivity_summary.md
这是文字版实验摘要。里面已经直接概括了：
实验目的；
判断稳定性的规则；
DFlow 在三个 rho 下的前五名单；
DGap 在三个 rho 下的前五名单；
以及哪些特征在三个 rho 下始终稳定进入前五。

如果你想把这些文件理解成一句话：
full_results 看全量明细，
top5_summary 看前五名单，
stability_summary 看稳定程度，
三类热力图分别看“关联度变化”“排名变化”“前五稳定性”，
summary.md 看文字结论。

---
# GRA 分辨系数敏感性实验摘要

- 实验目的：改变灰色关联分析分辨系数 rho=0.3、0.5、0.7，观察 DFlow 与 DGap 的前五特征排序是否稳定。
- 判断原则：若某特征在 3 个 rho 取值下均进入前五，则认为该特征具有较强稳定性。

## DFlow 前五特征对比
- rho=0.3：ASM_A_MeanT, ALM_A_MeanT, ALD_A_SDevT, CLD_C_SDevT, ILD_I_SDevT
- rho=0.5：ASM_A_MeanT, ALM_A_MeanT, ALD_A_SDevT, CLD_C_SDevT, ILD_I_SDevT
- rho=0.7：ASM_A_MeanT, ALM_A_MeanT, ALD_A_SDevT, CLD_C_SDevT, ILD_I_SDevT
- 在三个 rho 下均进入前五的稳定特征：ASM_A_MeanT, ALM_A_MeanT, ALD_A_SDevT, CLD_C_SDevT, ILD_I_SDevT。

## DGap 前五特征对比
- rho=0.3：CSM_C_MeanT, CLM_C_MeanT, BLD_B_SDevT, CSD_C_SDevT, ILD_I_SDevT
- rho=0.5：CSM_C_MeanT, CLM_C_MeanT, BLD_B_SDevT, ILD_I_SDevT, CSD_C_SDevT
- rho=0.7：CSM_C_MeanT, CLM_C_MeanT, BLD_B_SDevT, ILD_I_SDevT, CLD_C_SDevT
- 在三个 rho 下均进入前五的稳定特征：CSM_C_MeanT, CLM_C_MeanT, BLD_B_SDevT, ILD_I_SDevT。

