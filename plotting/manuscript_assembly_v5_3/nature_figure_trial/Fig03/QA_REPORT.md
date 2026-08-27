# Fig3 Nature-width reference: QA

- Status: PASS after 600 dpi PDF render inspection.
- Figure width: 183 mm.
- Figure height: 108 mm.
- PDF: one page with embedded Arial/Arial Bold fonts plus embedded DejaVu Sans
  math glyphs for the schematic symbols.
- Raster output: 4322 px wide at 600 dpi; the independent PDF render may differ by one pixel because of page-point rounding.
- Vector output: editable-text SVG on the 183 x 108 mm canvas.
- Grouping invariant: b, d, and f do not wrap.
- Panel f integrity: four axes are independently redrawn from the source-data table; no composite raster is used.
- Geometry audit: every quantitative axis has a 24 mm physical y-axis height; d1 and d2 are 42 mm wide and d1, d2, and e fill one row including their external tick labels.
- Panel-order audit: natural reading order restored as `a+b+c / d+e / f` without moving any data axes.
- Panel-a audit: native vector schematic installed at 34 x 20.5 mm and shifted
  3.5 mm left from the standard first-column origin; all text and response traces
  remain inside the canvas and clear of panel b.
- Source-data audit: every mapped input exists; no quantitative panel uses a legacy raster.
- Collision audit: PASS; panel letters, legends, long descriptor labels, confidence intervals, and n labels remain separated.
