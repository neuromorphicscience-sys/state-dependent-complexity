# Final Supplementary Figure blueprint

Status: **LOCKED FOR REFERENCE-LAYOUT GENERATION**  
Date: 2026-08-25

This blueprint reorganizes the 57 existing supplementary image assets after the
freezing of main Fig.1-6 and ED Fig.1-6. It controls scientific allocation and
panel order; it does not freeze graphical layout or captions.

## Allocation summary

- Existing image assets audited: **57**
- Retained as standalone axes: **26**
- Retained and merged into multi-axis panels: **23**
- Converted to a supplementary table: **1**
- Removed as frozen-figure duplicates: **7**
- Image assets entering final SI1-SI8: **49**
- Final visible panel letters across SI1-SI8: **36**

The seven removed assets are the complete former S08 block. Their source data and
scientific roles are already represented in frozen ED3; repeating them in SI would
be redundant. No other SI source-data file is an exact file-level duplicate of a
frozen main or ED source table.

## SI1 - SynPhys cohort and intrinsic-complexity stratification

**Scientific claim:** Intrinsic-complexity assignments are transparently supported
across PFC subtypes, Cre classes and cortical layers, with cohort size made explicit
and zero-variance structural fields reported separately.

Panel order:

1. **a** - PFC subtype structural effects (asset 01).
2. **b** - Complexity stratification by Cre type and cortical layer, two adjacent
   microplots (assets 45-46).
3. **c** - PFC subtype sample sizes (asset 02).

Moved to **Supplementary Table 1**: zero-variance structural-variable audit
(asset 03).

## SI2 - SynPhys local-circuit robustness and reductionist boundary

**Scientific claim:** SynPhys local-circuit associations survive repeated holdouts,
but cell identity absorbs much of the apparent mapping and the remaining continuous
local effect is bounded; intrinsic-complexity value is therefore not reducible to a
single local circuit statistic.

Panel order:

1. **a** - Local-circuit conceptual boundary map (asset 44).
2. **b** - Connection repeated-holdout robustness (asset 47).
3. **c** - Primary and conservative identity-absorption dumbbells, two adjacent
   microplots (assets 48-49).
4. **d** - Repeated experiment-holdout continuous effect (asset 50).

## SI3 - Steinmetz behavioral relevance of dynamical influence

**Scientific claim:** After session-duplication control, dynamical influence carries
behaviorally relevant information across choice, outcome and engagement, while
remaining distinct from conventional predictive importance.

Panel order:

1. **a** - Session duplicate audit (asset 04).
2. **b** - Predictive contribution to choice, outcome and engagement, three adjacent
   microplots (assets 05-07).
3. **c** - Predictive importance versus dynamical influence (asset 08).

## SI4 - Adaptive-lag and latent-model diagnostics

**Scientific claim:** Lag selection is empirically state structured and improves
held-out latent-model quality without trivially manufacturing residual state
stability.

Panel order:

1. **a** - Selected-lag distribution (asset 09).
2. **b** - Lag selection by state (asset 10).
3. **c** - Raw state-pair matrix (asset 11).
4. **d** - Model-quality improvement distribution (asset 12).
5. **e** - Selected lag versus held-out quality (asset 13).
6. **f** - Model quality versus residual stability (asset 14).

## SI5 - OpenScope cohort and analysis-validity audit

**Scientific claim:** OpenScope state analyses are supported by adequate half-model
predictive validity, a quantified low-dimensional structure, mouse-wise state
balance and transparent unit/trial counts.

Panel order:

1. **a** - Half-model minimum predictive validity (asset 38).
2. **b** - Dominant low-dimensional mode (asset 39).
3. **c** - State-fraction balance (asset 40).
4. **d** - Units, analysis trials and discovery-versus-final trial counts, three
   adjacent microplots (assets 41-43).

## SI6 - OpenScope coefficient-landscape controls

**Scientific claim:** The home-versus-cross coefficient-landscape advantage is
mouse consistent, persists after activity and selectivity residualization, and
extends to the real-edge definition.

Panel order:

1. **a** - Within- versus cross-state coefficient-landscape similarity (asset 15).
2. **b** - Mouse-wise coefficient-landscape effect (asset 16).
3. **c** - Raw versus controlled coefficient landscape (asset 17).
4. **d** - Activity-only and activity-plus-selectivity residualization, two adjacent
   microplots (assets 18-19).
5. **e** - Real-edge coefficient-landscape extension (asset 25).

## SI7 - OpenScope coefficient-landscape inferential sensitivity

**Scientific claim:** The coefficient-landscape state imprint is stable across
regularization choices, residualized definitions, high-precision permutation,
leave-one-mouse-out analysis and repeated-split precision.

Panel order:

1. **a** - Raw landscape regularization sensitivity for effect, within-state and
   cross-state metrics, three adjacent microplots (assets 20-22).
2. **b** - Regularization under activity and activity-plus-selectivity residualized
   definitions, two adjacent microplots (assets 23-24).
3. **c** - High-precision permutation boundary (asset 26).
4. **d** - Leave-one-mouse-out coefficient-landscape effect (asset 27).
5. **e** - Repeated-split instability and precision (asset 28).

## SI8 - OpenScope top-unit and top-set robustness

**Scientific claim:** A discrete top-unit/top-set representation independently
recovers the home-state advantage across budgets, overlap definitions,
regularization, formal permutation and leave-one-mouse-out analysis.

Panel order:

1. **a** - Aggregate and mouse-wise top-unit home-versus-cross transfer, two adjacent
   microplots (assets 29-30).
2. **b** - Top-unit budget sensitivity in AUC and balanced accuracy, two adjacent
   microplots (assets 31-32).
3. **c** - Top-set overlap and mouse-wise delta-Jaccard, two adjacent microplots
   (assets 33-34).
4. **d** - Regularization sensitivity of delta-Jaccard (asset 35).
5. **e** - High-precision permutation boundary for delta-Jaccard (asset 36).
6. **f** - Leave-one-mouse-out top-set effect (asset 37).

## Locked implementation rules

- SI numbering and panel order above are locked before reference-layout generation.
- Multi-microplot panels must remain on one row and may not wrap.
- The existing Nature figure style remains authoritative: 183 mm width, Arial,
  fixed physical panel-label offsets, explicit axis titles and local legends.
- Default inter-row data-axis gap is 13 mm; deviations require a documented label or
  heatmap-clearance reason.
- Every final panel must map to its original source-data file(s).
- Supplementary Table 1 must retain the source-data provenance of asset 03.
- Former S08 assets remain archived in place but are excluded from final SI exports.
