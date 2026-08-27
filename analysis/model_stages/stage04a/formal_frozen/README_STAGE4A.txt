# Stage 4A Patch

Copy these files into:

D:\Research\Neural Science

This does NOT replace Stage 4 code.

It only adds:
- configs\stage4a.json
- run_stage4a.ps1

Stage 4A scope:
- topology: scale-free only
- regimes: 3
- K values: 6
- graph seeds: 2

Total optimization tasks:
3 x 6 x 2 = 36

Formal numerical settings are unchanged:
- dt = 0.05 ms
- duration = 2500 ms
- warmup = 500 ms
- population = 4096
- elite = 512
- generations = 6
- random injection = 512
- swap mutations = 2

Frozen-result reproducibility:
- Use configs\stage4a.json for the formal Stage 4A discovery run.
- configs\stage4.json is a legacy/general Stage 4 configuration and must not be
  used to reproduce the frozen Stage 4A results.
- The formal checkpoints contain a 4096 x 256 population tensor, retain 512
  elites, generate 3072 two-swap offspring and inject 512 random masks per
  generation.
- The archived runtime config is preserved in stage4_complete_results.tar.gz.

Run:

powershell -ExecutionPolicy Bypass -File .\run_stage4a.ps1

Output:

results_stage4a\checkpoints\
results_stage4a\tasks\
results_stage4a\stage4_discovery_methods.csv

Note:
The current generic analyze_stage4.py is hard-coded to configs\stage4.json/results_stage4.
For Stage 4A, first complete the 36 discovery tasks.
Then upload the Stage 4A task CSVs or consolidated output and analyze before launching Stage 4B.
