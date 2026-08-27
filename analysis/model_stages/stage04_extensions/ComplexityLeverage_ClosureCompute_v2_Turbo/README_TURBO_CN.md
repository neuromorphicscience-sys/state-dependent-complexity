# Complexity–Leverage Closure Compute v2 Turbo

本版本**不改变 Stage4 科学定义**，只改变执行布局。

关键修正：每个 cross-state task 实际只有 12 个 source masks，因此 v1 的 `mask-batch=12` 已经是全 mask batch；真正的 GPU 吞吐瓶颈是 24 个 fresh noise seeds 被 Python 串行执行。v2 将独立 noise seed 与 12 个 masks 组成联合 GPU batch，并将 27 个独立 task 分片到多个进程。

默认推荐：`6 task shards × seed_batch=4 × 12 masks`。每个 task 仍使用完整 fresh seeds 7601–7624；所有 Stage4 equations、topology、regime parameters、common-noise design、score metrics、统计分析与 v1 保持一致。

新增：
- `stage4_gpu_simulator_turbo.py`: seed×mask joint batching；
- `stage4_cross_state_transfer_turbo.py`: task sharding、resume、严格 finalize；
- `run_stage4_turbo_6way.sh`: 6 路一键运行，完成后自动 finalize + functional degeneracy；
- `status_turbo.sh`: 查看 27 task 进度和 GPU；
- `tests/test_stage4_turbo_parity.py`: legacy 与 turbo 数值一致性 smoke test。

输出目录：
`/data/coding/NeuralScience/results_complexity_leverage_closure_v2_turbo/`

若服务器中断，重新执行同一 master 命令即可；完整 task 会根据 `done.json` 自动跳过。
