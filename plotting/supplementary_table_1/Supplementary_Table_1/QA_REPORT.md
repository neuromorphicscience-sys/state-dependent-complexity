# Supplementary Table 1 QA report

- Status: PASS for writing-stage delivery.
- Source schema: 26 rows x 8 columns; unchanged source CSV retained.
- Main table schema: 26 variables x 10 presentation columns.
- Zero-variance audit: exactly two fields, `dendrite_length` and
  `dendrite_nodes`; both are consistently marked in XLSX, DOCX and PDF.
- Identifier boundary: `archive_id` and `cell_id` are marked as identifiers.
- Categorical boundary: `projection_subtype` is marked as a categorical code.
- PDF: one A4 landscape page; visual render checked with no clipping,
  overlap, broken glyphs or unreadable rows.
- XLSX: three worksheets (`Table 1`, `Raw source`, `Readme`), frozen headers,
  filterable main/raw tables and landscape print setup.
- DOCX: A4 landscape section; 27-row x 10-column table with repeat-header
  markup and fixed column widths. Structural validation passed. Native Word
  pagination could not be rendered in this environment because Word/LibreOffice
  is unavailable; the reviewed PDF is the authoritative layout reference.
- Provenance: artifact hashes and the repository-relative source path are stored
  in `DELIVERY_MANIFEST.json`.
