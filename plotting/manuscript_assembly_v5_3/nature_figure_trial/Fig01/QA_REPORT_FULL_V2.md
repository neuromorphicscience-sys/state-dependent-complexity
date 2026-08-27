# Full Fig1 V5.3 final-spec redesign: QA report

- Status: PASS
- Supersedes: the rejected equal-size/equal-column drafts
- Backend: Python / Matplotlib only
- Combined physical size: 183 x 118 mm
- Nature constraint: double-column 183 mm width; well below the 170 mm maximum height
- Layout: two rows by three columns for article placement
- Composite spacing: row-to-row axes gap reduced from 26 to 20 mm; columns tightened to a regular 58 mm placement rhythm without changing axes-body sizes
- Combined axis geometry: a/c/d/e/f 45 x 36 mm; b 36 x 36 mm; all y-axis heights 36 mm
- Leaf axis geometry retained: a 70 x 42 mm; b 46 x 46 mm; c-f 58 x 42 mm
- Colorbar: separate vertical strip to the right of b; it does not resize the heatmap body
- Source data: all six frozen V5.3 Fig1 source-data tables
- Raster output: 600 dpi PNG and TIFF
- Vector output: editable-text SVG and one-page PDF with embedded Arial fonts
- Color semantics: sparse teal, transition dense blue, transition mid purple in every panel
- Heatmap: blue-white-red scale centered exactly at zero; subordinate external vertical colorbar
- Legends: compact in-axes legends throughout; a at upper left, c at lower right, and d/e at upper left
- Legend overlap audit: PASS; c/d/e legends occupy data-sparse regions and do not cover points, lines, confidence intervals, or paired trajectories
- Leaf outputs have no legend, a-f letters, or panel titles
- Visual cleanup: restrained line weights, solid markers, lighter confidence ribbons, compact panel-specific legends, and no in-panel audit text
- PDF render inspection: no clipped panel labels, overlapping colorbar ticks, broken glyphs, or panel-to-panel text intrusion

Statistical units and sample sizes remain recorded in the figure contract and source-data manifest rather than being placed inside every data region.
