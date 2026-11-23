# C1 synthetic cis-regulatory element design: manuscript code and source tables

Repository: https://github.com/nazbuhn/Tex-prog-like_synthetic_cre_design

Code, sequence files and supporting result tables for the C1 manuscript.
Zenodo deposit DOI supplied by the authors: https://doi.org/10.5281/zenodo.24480177
Public accessibility of the archive files has not yet been verified during packaging. Files are
existing analysis records; packaging ran no model inference, training, sequence
optimization or new shuffle generation.

## Deposit components

The code component contains scientific scripts, configurations, sequence FASTAs,
backgrounds, population labels, exact peak splits, saved summaries and motif
outputs. The source-data component contains measured population-count arrays,
saved predictions and original design input/control arrays. The model component
contains the trained checkpoint, reconstruction code and output metadata.

Supplementary File 1 contains 2,504 starting/candidate FASTA records representing
1,252 designs. Supplementary File 2 contains 20,032 saved audit-control sequences.
These controls are distinct from STREME's internally generated negative sequences.

## Finding the requested data

| Location | Contents |
|---|---|
| `data/sequences/` | Starting, selected candidate, native endpoint and audit-control FASTAs for each insert length |
| `data/design_metadata/designs.tsv` | Length-qualified public IDs, original IDs, parent IDs, offsets, starting arm and retention fractions |
| `data/backgrounds/backgrounds_2048bp.fasta` | Actual background bases, verified identical across all three libraries |
| `data/backgrounds/context_metadata.tsv` | Two optimization, four selection and eight final audit backgrounds |
| `data/backgrounds/background_coordinates.tsv` | Source peak indices and coordinates; derived model-input boundaries |
| `data/regions/peak_splits.tsv.gz` | All 439,167 measured peaks with exact training/validation/test assignments |
| `data/regions/TRAINING_SEEDS.tsv.gz` | Original training-region seed nominations and their coordinates |
| `data/populations/` | Eleven outputs, cell labels, cell counts, study composition, scaling factors and macro-AP eligibility |
| `data/random_seeds.tsv` | Recorded training, optimization, interpretation and STREME seeds |
| `results/designs/` | Original design configurations, saved predictions and selection/audit records for each insert length |
| `results/model_evaluation/` | Selected-epoch history, primary test metrics and all-comparator sensitivity analysis |
| `results/motif_analysis/` | IG/EIG settings, raw-pattern counts, representative profiles, reference matches, RNA-supported candidates, FIMO and mutation summaries |
| `results/streme_discovery/`, `results/streme_jaspar2026/` | STREME discovery commands/outputs and JASPAR 2026 Tomtom comparisons |
| `code/training/` | Exact executed training source, including the repaired loss, verified against its saved source manifest |
| `code/sequence_200bp/`, `code/sequence_500bp/`, `code/sequence_1000bp/` | Original optimization/evaluation scripts, verified against saved source manifests |
| `code/motif_analysis/` | Current executed motif workflow, including the final representative-export adapter |
| `code/rna_preprocessing/` | Seven checksum-verified RNA QC/export scripts and original selection policies |
| `code/integration/` | Recovered RNA/ATAC Harmony code and coordinate verification |
| `data/embeddings/` | Paired coordinates and batch covariates for the 111,347 plotted cells |
| `code/upstream_preparation_drivers/` | Recovered linkage/ATAC preparation drivers; see the scope limit below |
| `provenance/` | Identity checks, source manifests, metric recomputation audit and historical environment records |

## Inspecting and reproducing saved results

Verify the repository files:

```bash
python3 verify_deposit.py
```

After extracting the source-data archive, reproduce validation/test metrics on a
CPU using NumPy and the original recorded scoring functions:

```bash
python3 verify_saved_results.py --source-data /path/to/C1_SOURCE_DATA
```

No trained model is loaded for this check. Reported C1 test AP is 0.2418648128
(primary) and 0.2285057165 (all-comparator sensitivity). Positive counts and all
reported top-20/50/100/500 hit counts reproduce. The largest absolute difference
between direct CPU AP recalculation and any saved AP is 7.49e-9; the original
reported results are preserved. The verifier reports this difference and rejects
AP discrepancies of 1e-7 or larger rather than silently overwriting values.

The saved NPZ prediction axes are design × variant × background × output. Read
`design_ids`, `variants`, `context_ids`, `class_names` and `target_index` from the
same NPZ. Portable compressed tables expose all 11 outputs for initial, selected
and endpoint variants. The original NPZ also includes all 16 audit-control
variants per design. See `DATA_DICTIONARY.md`.

## Model identity and reconstruction

The manuscript C1 model is the selected epoch-8 checkpoint with 11 outputs.
Its exact SHA256 is:

`2aaccf0606f3832e8a768602298ab91d977ea0831dc31bb89360e8d6a3750e48`

The checkpoint and reconstruction instructions are prepared in a separate model
archive associated with the Zenodo deposit; public file access has not yet been verified. Archived training/optimization sources contain historical options
for other panels or secondary-model evaluations. Their presence does not mean
that every option was used as the manuscript's optimization objective. Read the
library-specific source and saved settings together. Original source has not been
rewritten to conceal those options.

The recorded training environment used CREsted 1.10.0, Keras 3.15.1 and PyTorch
2.13.0+cu130. The recorded motif-runtime versions include modisco-lite 3.0.0.
Stage-specific version records are provided; a current environment query is
identified as such and is not represented as a complete historical lockfile.
Fresh-environment model reconstruction/inference has not been executed during
packaging. Original HPC scripts retain their recorded paths and require path
configuration before reuse; they are not advertised as a portable one-command
end-to-end pipeline.

## Backgrounds, split and interpretation limits

All 14 background sequences match byte-for-byte across the 200/500/1000-bp inputs.
They comprise two optimization, four selection and eight final audit backgrounds,
all from training chromosomes. Their distinct roles must be retained. They are
not biological replicates or presumed neutral genomic DNA.

The exact split contains 356,938 training, 45,032 validation (chr8/chr10) and 37,197
test (chr9/chr18) peaks. Overlap with original Borzoi pretraining was not assessed.
The peak-split table contains original measured 500-bp peak coordinates. The
additional 2,048-bp model-input boundaries are derived by centering the recorded
input width on those peaks; these columns are explicitly labeled separately.
Coordinates are zero-based with an exclusive end, in mm10.

Raw 1,000-bp INPUTS contains 256 planned starting sequences. Its completed
EVALUATION_SEQUENCES array contains 196 designs. Those designs were aligned by
the original plan indices before exporting; the first 196 rows were not assumed
without checking. No missing designs were generated.

Saved audit-control sequences are distinct from STREME's internally generated
negative sequences. The latter exact FASTA was not found. Original STREME
commands, software version and seed are supplied, without claiming that a new
set of negative sequences is the original set.

The sequence designs and motif interpretations remain computational predictions,
not experimental demonstrations of TF binding, enhancer activity or cell-state
specificity.

## Source data and scope

Source sequencing: GEO GSE230363, GSE267584, GSE255042 and GSE290618. Genome: mm10.
Motif reference: JASPAR 2026 CORE vertebrates, nonredundant profiles. Reference
URLs and checksums are recorded in the motif reference manifest.

The priority sequence-design/model files, seven authenticated RNA QC/export
scripts, and recovered RNA/ATAC integration code are supplied. All 111,347 final
cohort cells link to the original RNA export. RNA Harmony source matches the
completed run's source checksum; the RNA coordinates match at stored float32
precision. ATAC coordinates match exactly, but its recovered script lacks a
historical source-checksum receipt. See `code/integration/README.md` and
`provenance/INTEGRATION_RECOVERY.json` for this distinction, batch covariates,
and the saved ATAC run's later plotting error. Upstream preparation drivers and
the common-matrix manifest are also provided. This deposit does not establish
that every upstream annotation script has been authenticated.

## Release and citation

Citation metadata and the repository URL are supplied in `CITATION.cff`.
The Zenodo DOI is 10.5281/zenodo.24480177. The versioned code release and licensing remain to be finalized.
Code, source data and checkpoint components are separate, with their own checksums.

Personal transfer instructions, status utilities and author-only packaging notes
are omitted. Scientific source files retain their original bytes. Package-level
`BUNDLE_SHA256.json` files describe the distributed subsets; historical full-source
manifests are preserved separately under `provenance/`. See
`provenance/PUBLICATION_CURATION.json` for the omitted-file list.
