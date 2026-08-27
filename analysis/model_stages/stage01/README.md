# Neural Science — Stage 1 GPU Discovery Harness

目标：在 HH–LIF 异质网络中寻找可重复、可推广、可理论化的节律涌现规律。

第一阶段不是为了“做出振荡”，而是为了系统搜索：
- critical transition
- minimal HH fraction
- non-monotonic robustness optimum
- topology dependence
- rare / multirhythmic regimes

## 环境

建议：
- Windows 11
- Python 3.11+
- NVIDIA GPU
- PyTorch 2.3+ with CUDA

安装：

```powershell
cd "D:\Research\Neural Science"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

检查 GPU：

```powershell
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

## 先跑 smoke test

```powershell
python scripts\run_sweep.py --config configs\smoke.json
```

结果：
- `results/runs.csv`
- `results/candidates.csv`
- 每批次的 checkpoint / metadata

## 正式第一轮

```powershell
python scripts\run_sweep.py --config configs\stage1_baseline.json
```

## 设计原则

每个 GPU batch 同时仿真多个独立网络 realization。
网络中一部分节点为 Hodgkin–Huxley，另一部分为 LIF。
当前版本使用有向 Erdos–Renyi 拓扑作为第一轮基线，后续 Stage 1B 再加入
small-world / scale-free / modular / motif-constrained topology。

为了让第一轮尽快验证“有没有值得继续的现象”，暂时控制变量数量，重点扫：
- HH fraction
- coupling strength
- connection probability
- noise
- random seed

自动计算：
- population firing rate
- dominant frequency
- spectral concentration
- synchrony proxy
- ISI CV
- burstiness
- silent fraction
- rhythm score

候选规则会优先记录：
- rhythm_score 高
- 中间 HH fraction 优于两端
- 强节律但低 HH fraction
- 邻近参数发生大幅跃迁

## 注意

这是 discovery harness v0.1，目的是先完成 Gate 1。
不要一开始把参数空间扩到十几个维度。
第一轮若没有发现可重复的 transition / optimum / sparse-HH effect，
应修改模型和网络结构，而不是盲目扩大 sweep。
