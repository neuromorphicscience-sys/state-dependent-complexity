# Full Fig1 nature-figure trial contract

Core conclusion:
Collective network state dynamically reassigns the functional value of cellular-complexity allocation: gains depend on state and budget, optimized allocations lose value across states, and the corresponding home-state advantage persists across matched topologies and replicates.

Figure archetype:
Compact two-row, three-column manuscript composite. Same-type Cartesian panels share a composite-fit axes body; the heatmap remains square and receives a separate vertical colorbar on its right.

Target journal/output:
Nature-family main figure; editable SVG primary, PDF with embedded Arial fonts, 600 dpi PNG and TIFF.

Backend:
Python / Matplotlib only.

Final size:
Combined reference 183 mm x 118 mm. Composite axes bodies: panels a and c-f are 45 x 36 mm; panel b is 36 x 36 mm. All six panels therefore share a 36 mm physical y-axis height. The independent leaf panels retain the V5.3 source specification: a 70 x 42 mm, b 46 x 46 mm, and c-f 58 x 42 mm. Text is specified at final placed size.

Panel map:
- a: State-by-complexity-budget gain relative to the spectral baseline.
- b: Cross-state transfer difference relative to the target-state home allocation.
- c: Matched-topology crossover interaction across complexity budgets.
- d: Paired transition-mid and sparse home-state advantages.
- e: Home-versus-foreign score geometry.
- f: Replicate-level crossover-interaction distributions.

Evidence hierarchy:
- Hero evidence: panel a establishes state-by-budget dependence.
- Validation evidence: panels b-d establish transfer loss, crossover, and paired home advantage.
- Controls/robustness: panels e-f expose observation-level geometry and replicate-level distributions.

Statistics needed:
- Panel a uses archived means and archived 95% intervals without recomputation.
- Panel b uses archived target-home-centered transfer differences.
- Panel c shows every archived graph-seed interaction and its descriptive mean.
- Panel d preserves all nine matched pairs.
- Panel e shows all 72 archived observations.
- Panel f shows all 36 archived replicate-level interactions; box summaries are descriptive.

Source data needed:
The six frozen per-panel V5.3 source-data tables for Fig1.

Image-integrity notes:
- No raster manipulation or value interpolation.
- All panels and the combined SVG/PDF are rendered directly from source data.
- Each panel is also exported separately; the combined figure does not hide composite leaf panels.
- Leaf panels contain no panel letters and no panel titles; letters are added only in the combined figure.
- The state legend is placed inside panel a in the combined figure and omitted from the independent leaf panel.
- Panel b uses a subordinate external vertical colorbar on its right.
- Panels c, d, and e use compact in-axes legends placed in data-sparse regions: c at lower right, d and e at upper left; no legend overlaps data.

Reviewer risk:
- Panel a contains only two paired tasks per state-budget cell; broad intervals must remain visible.
- Panels c and d contain three graph seeds across three budgets; pairing must remain visible.
- Panel f is the main replication safeguard and must retain every replicate point.
- State colors must remain identical across all six panels; budget colors must not reuse state-identity colors.
