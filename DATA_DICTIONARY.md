# Data dictionary

## Identity and sequence stages

`insert_bp` and original `design_id` together identify a design. Original IDs
such as C1L_0001 recur across libraries. `public_design_id` and FASTA headers
include insert length. `arm` is the recorded natural/random starting arm;
`parent_id`, `offset`, `retention_fraction` and original `index` come from the
saved plan. Bases were not modified during export.

`initial` is the unoptimized input. `selected` is the candidate selected from
the ISE trajectory; it can differ from the final iteration. `native_endpoint`
is the native endpoint. `dinuc_01`–`dinuc_16` are the original per-candidate
controls used in design evaluation, not STREME's internally generated negatives.

The original 1,000-bp INPUTS array has 256 planned starts. The completed
196-design evaluation array is aligned using the saved original plan indices.

## Prediction arrays and units

`ALL_PROFILES_Borzoi.npz` contains `predictions` with axes design × variant ×
background × output, together with axis labels `design_ids`, `variants`,
`context_ids`, `class_names` and scalar `target_index`. These files require no
pickle loading. They include all saved control predictions as well as starting,
selected and endpoint predictions.

Predictions represent population-level ATAC fragment-overlap counts normalized
per million unique fragments, after removal of population scaling. They are not
probabilities or observed counts. Some original column names use FRPM; the new
portable long tables spell out the quantity. Text exports use nine significant
digits to retain saved float32 values.

## Population labels and scaling

`model_outputs.tsv` gives the saved output order, cell counts, population fragment
exposure denominators and multiplicative CREsted scaling factors. Exported cell
labels reproduce all 11 saved population counts. Coarse C1 takes precedence;
other cells use their fine resolution-1.0 group; smaller fine groups are pooled.
The study-composition table is descriptive, not a batch-confounding analysis.
The five macro-AP outputs are explicitly marked. All 11 outputs entered training.

Primary validation and test comparisons exclude OTHER_SMALL_ATAC. Sequence
optimization/design evaluation and the saved sensitivity analysis include it.
The positive threshold of at least 10 fragments applies to raw measured fragment
counts aggregated across cells in the target population.

## Backgrounds and coordinates

`fresh_01`–`fresh_14` are respectively two optimization, four selection and eight
final audit backgrounds. `global_index` is the zero-based row in the original
region/sequence arrays. Exact background bases are verified identical across
all three libraries. Every background belongs to the training split.

The peak-split table preserves all original 500-bp mm10 peak coordinates and
exact split codes (0 training, 1 validation, 2 test). Coordinates are zero-based,
end-exclusive. Centered 2,048-bp model-input boundaries are derived as peak start
minus 774 and peak end plus 774, and are labeled separately from the measured
peak boundaries. No reference sequence was regenerated during packaging.

The parent-region-index table preserves the original seed-array ordering and
its genomic indices. It does not invent a mapping from an arbitrary parent-name
suffix to a genomic coordinate. Original training-seed nominations are supplied
separately with rank, region and support information.

## Table 1 calculation

For each design/background/stage, divide predicted C1 accessibility by the
maximum prediction among all ten non-C1 outputs. Calculate each design's median
across eight audit backgrounds, then the median across designs within each
insert-length/starting-arm group. The fold change is the optimized group median
divided by the initial group median, not a median of individual fold changes.
The all-eight count requires a ratio at least two in every audit background.
Exact results recalculated from saved predictions are supplied; intermediate
values were not rounded.

## File identity and numerical validation

SHA256SUMS files cover the contents of each deposit component. Original source
manifests establish historical code identity for the training and optimization
packages. Recalculated AP agrees within 7.49e-9 with saved values; positive counts
and top-k hit counts reproduce. Original reported metrics remain unchanged.
No model inference or new control generation was performed during packaging.
