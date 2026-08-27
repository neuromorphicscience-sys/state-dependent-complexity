# Full Fig1 nature-figure trial: QA report

- Status: PASS
- Backend: Python / Matplotlib only
- Frozen source data: all six V5.3 Fig1 source-data CSVs
- Combined physical size: 180 x 145 mm
- Leaf panels: six independent one-panel exports; no hidden multi-plot leaf composite
- Raster exports: PNG and LZW-compressed TIFF, 600 dpi
- Vector exports: editable-text SVG and one-page PDF
- PDF fonts: embedded Arial and Arial Bold as Type0 fonts
- State palette: sparse `#2A9D8F`, transition dense `#38598C`, transition mid `#8B7DAA`
- Heatmap palette: `#2166AC` - `#F7F7F7` - `#B2182B`, centered exactly at zero
- Budget palette: independent neutral blue-grey sequence, avoiding conflict with state identity
- Source-data coverage: a 18 rows, b 9 rows, c 9 rows, d 9 rows, e 72 rows, f 36 rows
- Visual inspection: final-size PDF and all six leaf PDFs rendered without clipped text, overlapping labels, broken glyphs, or colorbar overflow
- Statistical integrity: archived values retained; only descriptive means and standard box summaries are calculated where required for display

Reviewer-facing limitations retained in the figure:

- Panel a has two paired tasks per state-budget cell.
- Panels c and d contain three graph seeds across three budgets.
- Panel f retains every replicate-level point and is the main replication safeguard.

The trial is isolated from the frozen V5.3 atlas and does not overwrite existing manuscript panels or reference figures.
