# Neural Science Panel Suite V5.3 — Unified Redraw

This package is the V5.3 redraw upgrade requested after reviewing V5.2 outputs.
Its purpose is to regenerate **all panel-level figures** with one consistent rendering contract:

- use the cleaner color language from the preferred reference panels
- force **filled markers** instead of hollow circles
- keep **physical x/y axis lengths consistent** (not just outer image height)
- shrink overly large statistics text / annotations
- improve heatmap colors and colorbar styling
- preserve **all stages and all scientific outputs**, especially the high-value datasets:
  - Stage1 / Stage1B / Stage2 / Stage3
  - Stage4A / Stage4 integrated
  - Stage5A / Stage5B / closure extensions
  - Stage5C SynPhys (integrated + narrative + atlas)
  - OpenScope

## What V5.3 changes compared with V5.2

1. **Rendering only** — no scientific calculations are changed.
2. **Preferred palette** anchored on teal / blue / purple / orange.
3. **Filled markers** are enforced in the save hook.
4. **Heavy text is clamped** to smaller manuscript-safe annotation sizes.
5. **Heatmaps** are normalized to a cleaner sequential or diverging map.
6. **Physical axis-body normalization** remains active.

## How to run

From your project root (example: `D:\Research\Neural Science`):

```bash
python build_panel_suite_v5_3.py --root "D:\Research\Neural Science"
```

Optional examples:

```bash
# Run only early-stage redraw jobs
python build_panel_suite_v5_3.py --root "D:\Research\Neural Science" --stages stage1 stage1b stage2 stage3

# Run everything and stop on first hard failure
python build_panel_suite_v5_3.py --root "D:\Research\Neural Science" --fail-fast
```

## Output location

The suite writes a run bundle under:

```text
<root>/plot/Panel_Suite_V5_3/
```

And it redraws the actual stage outputs under the corresponding `plot/Stage*` folders, just like V5.2.

## Important scope note

V5.3 is for **panel regeneration**, not final multi-panel manuscript assembly.
That is intentional: you said the immediate goal is to get all leaf panels clean, consistent,
and suitable for manual main-figure assembly.

## Key files

- `build_panel_suite_v5_3.py` — full orchestration script
- `runtime/nature_style_v53.py` — unified style contract
- `runtime/axis_physical_hook.py` — save-time physical axis normalization + artist cleanup
- `config/COLOR_CONTRACT_V5_3.json` — palette contract
- `config/PHYSICAL_AXIS_CONTRACT_V5_3.csv` — target physical axis-body sizes

