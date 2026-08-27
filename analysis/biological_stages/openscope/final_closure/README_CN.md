# OpenScope Final Closure v1

这是 OpenScope 分析的最后封箱包。

它不再探索任何新 endpoint，只完成两件事：

1. **高精度 fixed-split state-shuffle null**
   - 默认 2000 permutations / mouse-session
   - 12 个 mouse/session
   - 只补 headline：
     - `R_decoder`
     - `R_rho`
   - `delta_jaccard` 顺手得到，不增加模型拟合量
   - **不再为 supportive top-20% unit-set 做 2000 次昂贵 CV null**
   - 自动复用 v2.3 已经算过的 30 permutations
   - 先精确验证轻量 null engine 与 frozen v2.3 前 3 个 permutation 完全一致
   - 支持断点续算
   - session 间并行，默认 4 workers

2. **12-mouse formal biological closure**
   - 12-repeat mouse/session effect size
   - bootstrap 95% CI
   - exact sign-flip
   - Wilcoxon
   - exact binomial sign test
   - paired standardized effect size `dz`
   - leave-one-mouse-out
   - primary vs real-edge paired boundary
   - confound audit
   - estimator predictive-validity audit
   - 最终机器可读 adjudication

## 一个重要的统计修正

v2.3 的 state-shuffle null 固定的是 **repeat-0 discovery/final split**。

因此 Final Closure 的 permutation p-value 使用：

`repeat-0 observed group mean`

与同一个 fixed-split null 比较。

而正式 effect size 仍使用：

`12 repeats -> within-mouse/session average -> 12 mice inference`

这样 null 与 observed statistic 是严格匹配的，不再把 12-repeat mean 与 repeat-0 null 混在一起。

## 放置

解压到：

`D:\Research\Neural Science\plot\OpenScope_FinalClosure_v1`

无需复制 v2.3 结果或 cache。软件会自动读取：

`D:\Research\Neural Science\plot\OpenScope_Illusion_Analysis_v2_3`

并自动从 v2.3 / v2.2 / v2.1 中寻找已经完成的 cache。

## 可选自检

正式包已经做过 syntax/self-test，但本地也可运行：

```powershell
cd "D:\Research\Neural Science\plot\OpenScope_FinalClosure_v1"
python .\openscope_final_closure_v1.py --self-test
```

应看到：

`SELF_TEST lightweight null exactly reproduces frozen v2.3`
`SELF_TEST OK`

## 直接一次性正式跑

```powershell
cd "D:\Research\Neural Science\plot\OpenScope_FinalClosure_v1"

python .\openscope_final_closure_v1.py `
  --root "D:\Research\Neural Science" `
  --permutations 2000 `
  --workers 4
```

如果 CPU 很强，可以：

```powershell
python .\openscope_final_closure_v1.py `
  --root "D:\Research\Neural Science" `
  --permutations 2000 `
  --workers 6
```

不建议超过 6 workers，避免多个 500–900 unit session 同时拟合导致内存/CPU 争抢。

## 中途退出

直接重新执行同一条命令。

每个 session 都会把已完成 permutation 写入：

`OpenScope_FinalClosure_v1\null_sessions\`

不会从 0 重新开始。

## 最终重点文件

`D:\Research\Neural Science\plot\OpenScope_FinalClosure_v1\analysis`

其中最重要：

```text
00_OPENSCOPE_FINAL_CLOSURE_REPORT.md
FINAL_BIOLOGICAL_CLOSURE.json
formal_12mouse_endpoint_table.csv
formal_12mouse_statistics.csv
high_precision_permutation_summary.json
high_precision_hierarchical_null.csv
leave_one_mouse_out.csv
leave_one_mouse_out_summary.json
primary_vs_real_edge_paired_boundary.csv
primary_vs_real_edge_boundary_statistics.json
run_manifest.json
```

跑完后直接把整个 `analysis` 文件夹压成 ZIP 发回即可。

## 冻结规则

本包不允许：

- 改 state 定义
- 改 response window
- 改 discovery/final split
- 改 C=1 primary
- 改 20% supportive budget
- 因结果改变 endpoint
- 把 unit / repeat / area 当独立 mouse
- 把 predictive/functional leverage 写成 causal leverage

完成本包后，OpenScope 科学分析应封箱，进入统一 Figure architecture 与 manuscript。
