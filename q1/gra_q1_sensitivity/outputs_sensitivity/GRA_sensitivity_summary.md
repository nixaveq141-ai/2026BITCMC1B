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

