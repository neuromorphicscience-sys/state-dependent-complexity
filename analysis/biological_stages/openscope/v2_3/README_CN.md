# OpenScope Illusion 000248 — Formal Integrated Analysis v2.3

这版就是**一次性正式跑**的版本。没有 `--quick`。

## 为什么从 v2.2 升到 v2.3

v2.2 的 Quick 运行已经完成了 12/12 mouse/session 的真实数据读取，也给了我们一个必要的方法学 QC：

- IC–LC decoder 本身是有效的（Quick 中 discovery AUC 平均约 0.726）；
- 但 zero-to-training-mean 单神经元 ablation ranking 的 split/repeat reliability 太低，不适合作为主 leverage estimator；
- focal-area → other-area continuous Ridge bridge 的 held-out R² 在 Quick 中全部为负，因此不能把其派生的 cross-area leverage 当作有效 headline endpoint。

因此 v2.3 **不改变生物学问题**，而是把正式主检验冻结为“可靠性归一化的 decoder/coefficient landscape”分析。

详细依据见：`METHOD_QC_FREEZE_v2_3.md`。

---

# v2.3 的两个最核心科学问题

## 1. State-specific population code 能否跨状态泛化？

在 discovery trials 中分别训练 State 0 和 State 1 的 IC-vs-LC decoder：

\[
D_0, D_1.
\]

然后在 untouched final trials 上计算：

\[
A_{00},A_{01},A_{10},A_{11}.
\]

Primary metric：

\[
\boxed{
R_{decoder}=\frac{AUC_{00}+AUC_{11}-AUC_{01}-AUC_{10}}{2}
}
\]

若 \(R_{decoder}>0\)，说明 source-state decoder 在 home state 中比跨 state 更能泛化。

---

## 2. Leverage landscape 的跨状态变化是否超过普通 sampling variability？

v2.3 不再把“cross-state rank correlation 很低”本身当成 state reconfiguration。

每个 state 的 discovery trials 都被分成两个**互不重叠、exact-label balanced** halves：

\[
0A,0B,1A,1B.
\]

每个 half 单独拟合 standardized L2-logistic decoder，取

\[
|\beta_i|
\]

作为该模型里的 unit leverage proxy。

然后：

\[
\rho_{within}=\frac{\rho(0A,0B)+\rho(1A,1B)}{2},
\]

\[
\rho_{cross}=\mathrm{mean}\{\rho(0A,1A),\rho(0A,1B),\rho(0B,1A),\rho(0B,1B)\}.
\]

最终：

\[
\boxed{R_{\rho}=\rho_{within}-\rho_{cross}}
\]

这才真正回答：

> between-state shift 是否大于同一 state 下 estimator/sampling 自己的不稳定性？

Top-20% Jaccard overlap 同样做一套完全平行的分析。

---

# 另外一个与 Stage4 最接近的功能检验

状态内 discovery decoder 的 standardized coefficient magnitude 用来选择 source-state top units。

Primary budget 固定：

\[
k/N=20\%.
\]

同时预先保存：

- 5%
- 10%
- 20% **primary**
- 40%

然后在 target-state final trials 中**重新拟合 decoder weights**，只保留 source-state 选出来的 unit set。

因此测试的是：

> 一个 state 中识别出的高-value unit set，在另一个 state 中是否仍然同样有用？

Primary metric：

\[
\boxed{
R_{topset}=\frac{AUC_{00}+AUC_{11}-AUC_{01}-AUC_{10}}{2}
}
\]

这比直接跨状态搬运 decoder weights 更接近 Stage4 的 allocation-transfer 逻辑。

---

# v2.3 的 Primary state：先去掉 running + slow drift

为了避免把 recording 前半段/后半段或者 locomotion 简单叫作“collective neural state”，v2.3 的 primary state 在 **discovery trials only** 拟合：

\[
B_i(t)\sim 1+t+t^2+t^3+\log(1+running).
\]

然后：

1. 对所有 trial 应用 discovery-fit residualization；
2. discovery-fit per-neuron z-score；
3. 移除每个 trial 的 global population mean；
4. discovery-only PCA；
5. PC1 最大绝对 loading 定向为正，避免 PCA 符号任意翻转；
6. discovery median split 定义 State 0/1；
7. untouched final trials 只做投影，不重新拟合任何 state 参数。

同时额外跑 4 次 **raw-state comparator**，用来量化 time/running confound 被消除了多少。

---

# Same-exact-image biological support

v2.3 不再把那个 held-out R² 为负的 cross-area Ridge predictor 当 primary。

取而代之的是一个更直接、有效性更清楚的 same-input test：

分别只看：

- IC1
- IC2

在**完全相同外部图像**下，用 poststimulus population pattern 去 decode prestimulus state。

并且每个 trial 先移除 poststimulus global population mean。

因此这是：

\[
\boxed{
\text{same exact external image}\rightarrow
\text{state-dependent distributed population response}
}
\]

它不是 leverage estimator，但可以作为“same input, different collective state”的直接生物学支撑。

---

# Controls / sensitivity

正式版默认同时完成：

- matched real-edge `IRE1/IRE2 vs TRE1/TRE2` control：6 repeats/session；
- raw-state comparator：4 repeats/session；
- regularization sensitivity：`C = 0.25, 1, 4`，`C=1` primary；
- top-unit budget sensitivity：5/10/20/40%，20% primary；
- activity + area residualized landscape；
- activity + area + IC/LC selectivity residualized sensitivity；
- 30 fixed-split, exact-label-preserving state-shuffle null / session；
- mouse/session 始终是唯一独立 biological replicate。

---

# Cache：不需要重新提取 186 GB 数据

v2.3 会自动依次寻找：

1. `OpenScope_Illusion_Analysis_v2_3\cache`
2. `OpenScope_Illusion_Analysis_v2_2\cache`
3. `OpenScope_Illusion_Analysis_v2_1\cache`
4. `OpenScope_Illusion_Analysis_v2\cache`

只要找到 `done.json=COMPLETE + trials.csv + units_primary.csv + spike_counts.npz`，就直接复用。

因此你之前 12/12 session 已经提取好的 cache 不需要复制，也不需要重新读取原始 NWB。

---

# 安装/环境

继续使用你现在已经能跑 v2.2 的同一个 Python 环境即可。

如果需要：

```powershell
pip install -r .\requirements_openscope.txt
```

---

# 放置

解压到：

```text
D:\Research\Neural Science\plot\OpenScope_Illusion_Analysis_v2_3
```

---

# 直接一次性正式运行

不需要 Quick。

```powershell
cd "D:\Research\Neural Science\plot\OpenScope_Illusion_Analysis_v2_3"

python .\openscope_illusion_analysis_v2_3.py `
  --root "D:\Research\Neural Science"
```

默认正式参数：

- 12 primary repeats / mouse-session
- 6 real-edge control repeats / mouse-session
- 4 raw-state sensitivity repeats / mouse-session
- 30 state-shuffle nulls / mouse-session
- discovery/final = 60/40
- top 20% units primary
- logistic `C=1` primary
- no publication figures

---

# 运行中断可以直接续跑

每个 primary repeat 会落盘；`repeat_summary.csv` 是 checkpoint marker。

如果中途断电/终止，直接重新运行同一条命令即可。脚本会跳过已经完整写出的 repeat。

`done.json` 还保存 parameter signature；不同参数不会被误当成同一个正式运行。

---

# 最终最重要输出

```text
analysis\00_OPENSCOPE_V2_3_FINAL_REPORT.md
analysis\primary_adjudication_v2_3.json
analysis\run_manifest_v2_3.json
analysis\analysis_status.csv
analysis\session_primary_v23_summary.csv
analysis\topset_budget_sensitivity_inference.csv
analysis\landscape_regularization_sensitivity_inference.csv
analysis\session_raw_state_sensitivity.csv
analysis\state_confound_control_comparison.csv
analysis\session_real_edge_v23_control.csv
analysis\hierarchical_v23_state_shuffle_null.csv
```

每个 session 还保留：

```text
primary_v23_decoder_transfer.csv
primary_v23_topset_transfer.csv
primary_v23_landscape_summary.csv
primary_v23_landscape_unit_coefficients.csv
primary_v23_same_image_state_imprint.csv
primary_v23_state_scores.csv
primary_v23_state_shuffle_null.csv
stage_progress.csv
```

---

# 解释边界

v2.3 的 coefficient magnitude 是 standardized regularized decoder 中的 predictive/functional leverage proxy。

它不是因果权重，也不是突触强度。

只有当：

- split-half decoder 本身有 held-out predictive validity；
- `R_rho > 0`；
- top-set transfer / decoder transfer 至少方向一致；
- confound-controlled state 不再主要等同于 time/running；
- mouse/session-level replication 稳定；

我们才把 OpenScope 写成 positive biological closure。
