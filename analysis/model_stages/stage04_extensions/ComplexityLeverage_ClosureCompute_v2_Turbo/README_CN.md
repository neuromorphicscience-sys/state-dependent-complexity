# Complexity–Leverage Closure Compute v1

这是 Stage 1–5B 主线之后的 **三项定向 closure compute** 软件包。它不扩大一般性参数扫描，只回答目前最关键的三个逻辑缺口：

1. **Stage 4 cross-state transfer**：不同状态中优化得到的复杂度分配，在另一个状态中是否真正失去功能价值？
2. **Stage 4F functional degeneracy formalization**：把“多个不同 mask 得到相近功能”正式写成可量化的 near-optimal equivalence class。
3. **Allen Visual Behavior lag-aware latent dynamics refit**：用严格 blocked validation 选择动力学 lag，并在 untouched test block 上重新评估 latent-dynamics quality，解决当前 Allen one-step R² 偏弱的问题。

默认项目根目录为：

```text
/data/coding/NeuralScience
```

输出完全写入新目录，不覆盖 Stage 1–5B：

```text
/data/coding/NeuralScience/results_complexity_leverage_closure_v1
```

---

## 1. Stage 4 cross-state transfer

Stage4F 已经证明“mask 身份变化”并不显著超过 optimizer variability，因此不能再用 mask overlap 本身定义 state specificity。本包改用**功能迁移**定义。

对同一个 `graph_seed, k`，将状态 `s` 中独立优化得到的四个 mask 放到目标状态 `t` 中，用一套全新的 common-noise seeds 做 fresh reevaluation。

目标态中心化 transfer：

```text
G(s -> t) = mean Phi(A_s, t) - mean Phi(A_t, t)
```

其中对角线恒为 0，负值表示 foreign-state mask 在目标态中相对 home-state mask 发生功能损失。

最关键的 primary contrast 只比较：

```text
transition_mid  vs  sparse_drive
```

因为两者都使用 `connection_prob = 0.05`，同 seed 下拓扑完全相同，只改变动力学 operating regime。对每个 anchor 定义 crossover interaction：

```text
I = 1/2 * [Phi(A_mid, mid)-Phi(A_sparse, mid)
         + Phi(A_sparse, sparse)-Phi(A_mid, sparse)]
```

`I > 0` 才支持“状态会重权 dynamical-leverage landscape”。正式统计以 9 个独立 `(graph_seed, k)` anchors 为单位，使用 exact sign-flip test 和 anchor bootstrap CI；optimizer replicate 不被当作独立生物学/图实例。

`transition_dense` 使用 `p=0.10`，所以所有涉及 dense 的 cross-transfer 均自动标记为 **topology+dynamics composite**，不能被写成 pure state test。

本包直接冻结并复用 Stage4F closure archive 中的 `stage4_gpu_simulator.py`、`stage4_topology.py` 和 `stage4a.json` source snapshot，不重新定义模型。

默认使用全新的 24 个 validation seeds `7601–7624`，和 Stage4F 原来的 `6501–6524` 分离。

---

## 2. Functional degeneracy

定义经验 near-optimal allocation class：

```text
E_epsilon(G,s,k) = {A_r : Phi(A_r) >= max_j Phi(A_j) - epsilon}
```

并定义 sampled functional degeneracy：

```text
D_epsilon = mean_{A<B in E_epsilon} [1 - Jaccard(A,B)]
```

主阈值 `epsilon=0.02`，同时做 `0.01, 0.05` 敏感性分析。额外量化：

- sampled near-optimal solution count；
- union expansion `|union E|/k`；
- core fraction `|intersection E|/k`；
- union 内 node-selection binary entropy；
- score gap–mask distance geometry；
- low-budget (`k=8`) vs high-budget (`k=64`) 的 paired anchor comparison。

**边界条件**：每个条件只有 4 个独立 optimizer solutions，因此这是对 equivalence class 的经验抽样/下界，不得写成完整 combinatorial solution-space volume。

在已有 Stage4F archive 上，本脚本的 archive-only 自检应得到：

```text
conditions = 27
optimizer replicates / condition = 4
```

如果 Stage4 cross-state transfer 已完成，还会用第二套 fresh noise 的 home-state score 再做一次独立 degeneracy replication。

---

## 3. Allen Visual Behavior lag-aware refit

当前 Stage5B 的 Allen primary one-step latent model 比较弱。本包不根据最终 leverage 结果选择 lag，而采用严格的时间块 model selection：

```text
0–60%   training
60–80%  validation: 选择 lag 和 ridge alpha
80–100% untouched final test
```

默认候选：

```text
lag = 0.25, 0.5, 1.0, 2.0 s
alpha = 0.1, 1, 10
```

选择 criterion 仅为 validation latent-dynamics R²。选择后，在前 80% 上重新拟合 scaler/PCA/Ridge，并只在最后 20% 报告 final R²。

同时加入 persistence baseline：

```text
R²_persistence : Z(t+lag) ~= Z(t)
Delta R² = R²_model - R²_persistence
```

因此最终会同时回答：

- lag-aware model 是否提高绝对 held-out R²；
- 是否真正优于简单 persistence；
- quality-gated 后 same-cell leverage reconfiguration 是否仍存在；
- activity/loading residualization 后结果是否保留。

为了避免 test leakage，最终 `L_dyn` 的方向来自前 80% refit，activity amplitude 也只由前 80% 估计；最后 20% 只用于 model-quality evaluation。

---

# 推荐运行方式

先解压到云端项目根目录附近，例如：

```bash
cd /data/coding/NeuralScience
unzip ComplexityLeverage_ClosureCompute_v1.zip
cd ComplexityLeverage_ClosureCompute_v1
```

先做环境检查和 dry-run：

```bash
python scripts/check_env.py
python tests/test_core.py
python scripts/stage4_cross_state_transfer.py \
  --project-root /data/coding/NeuralScience \
  --stage4f-source auto \
  --dry-run
```

如果 dry-run 应输出：

```text
conditions = 27
source_masks = 108
transfer_tasks = 27
```

## 推荐：GPU 和 CPU 两路并行

终端 A（GPU，Stage4 transfer + degeneracy）：

```bash
cd /data/coding/NeuralScience/ComplexityLeverage_ClosureCompute_v1
nohup bash run_stage4_only.sh /data/coding/NeuralScience \
  > stage4_closure.log 2>&1 &
```

终端 B（CPU，Allen lag refit；默认 4 workers）：

```bash
cd /data/coding/NeuralScience/ComplexityLeverage_ClosureCompute_v1
nohup bash run_allen_only.sh /data/coding/NeuralScience \
  > allen_lag_refit.log 2>&1 &
```

查看进度：

```bash
bash status.sh /data/coding/NeuralScience
```

也可以顺序一次性运行：

```bash
nohup bash run_all.sh /data/coding/NeuralScience \
  > closure_all.log 2>&1 &
```

---

# Stage4F source 自动发现

程序按顺序寻找：

```text
/data/coding/NeuralScience/results_stage4f_closure/
/data/coding/NeuralScience/stage4f_closure_results.tar.gz
/data/coding/NeuralScience/results_stage4f_closure.tar.gz
```

如果都不在，可以显式传：

```bash
python scripts/stage4_cross_state_transfer.py \
  --project-root /data/coding/NeuralScience \
  --stage4f-source /你的路径/stage4f_closure_results.tar.gz
```

---

# Allen 数据默认路径

```text
/data/coding/NeuralScience/biological_data/stage5b_dynamic/allen_visual_behavior_2p_official_s3
```

如果位置不同：

```bash
python scripts/allen_lag_aware_refit.py \
  --project-root /data/coding/NeuralScience \
  --allen-root /你的Allen目录 \
  --workers 4
```

程序支持 per-experiment checkpoint，可中断重启，不会重复已成功 experiment。

---

# 最终输出

```text
results_complexity_leverage_closure_v1/
├── stage4_cross_state_transfer/
│   ├── analysis/
│   ├── figures/
│   └── tasks/
├── stage4_functional_degeneracy/
│   ├── analysis/
│   └── figures/
├── allen_lag_aware_refit/
│   ├── analysis/
│   ├── figures/
│   └── checkpoints/
├── closure_summary.json
└── closure_summary.md
```

所有图：

- 无标题；
- 单独子图；
- 600 dpi PNG；
- 同时输出可编辑 PDF；
- 沿用 Stage4/5 的 blue–teal–orange–purple 配色体系。

完成后：

```bash
bash collect_results.sh /data/coding/NeuralScience
```

会生成一个新的 `.tar.gz` 和 `.sha256`，把这个结果包下载给分析窗口即可。

---

# 这三项计算的判读规则

### Cross-state transfer

如果 primary `transition_mid ↔ sparse_drive` 的 anchor-level crossover interaction 明显 `>0`，并且 bootstrap CI 不跨 0 / exact sign-flip p 较小，则可以把主张升级为：

```text
State reweights the functional value of cellular complexity over a degenerate allocation landscape.
```

若不成立，也不是项目失败；那更支持“leverage backbone / functional degeneracy 很强，state-specific allocation 较弱”的模型。

### Functional degeneracy

如果低预算下 `D_epsilon`、union expansion、selection entropy 较高，同时 near-optimal count >1，则支持：

```text
Scarce intrinsic complexity admits multiple structurally distinct but functionally equivalent allocations.
```

### Allen lag-aware refit

最理想结果是：

```text
median held-out R² 明显提高
positive-R² fraction 提高
Delta R² vs persistence > 0
quality-gated leverage reconfiguration 保留
```

此时 Allen VBO 可以从“带 caveat 的观察结果”升级成更坚实的 state/context-resolved dynamical-leverage evidence。
