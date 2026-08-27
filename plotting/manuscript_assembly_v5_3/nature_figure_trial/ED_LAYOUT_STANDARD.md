# Extended Data layout standard

This contract is inherited from the frozen main Fig01 and applies to ED01-ED06.

- Full-width canvas: 183 mm (Nature double-column width).
- Inter-row gap: 13 mm, measured between adjacent data-axis rectangles.
- Panel labels: lowercase Arial Bold, 8 pt.
- Panel-label anchor: 8 mm left of the data-axis left edge and 2 mm above its top edge.
- Axis heights are equal within a figure unless the scientific chart type requires otherwise.
- Labels use fixed physical offsets, never proportional offsets, so their placement survives canvas changes.
- Legends stay local to their panels and must not alter the 13 mm row gap.
- Every exported PDF is re-rendered at 600 dpi before freezing.
