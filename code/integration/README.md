# RNA and ATAC integration source used for the displayed coordinates

RNA: `rna/rna_state_review.py` was recovered byte-for-byte from the original
saved installer. Its SHA256 matches `SOURCES_SHA256.json` in the completed run.
Configuration/reference files and recorded requirements also match that source
manifest. The released RNA results concern the `all_paired_CD8` branch; the source
also contains a separate TIL review branch. No personal installer or status/
upload instructions are included.

ATAC: `atac/run_atac_umap.py` was recovered from the saved ATAC Harmony directory.
Its saved coordinates match the paper's frozen ATAC coordinates exactly for all
111,347 plotted cells. No historical script-checksum receipt was found for this
ATAC run, so byte-level historical source authentication is not claimed. The
saved log confirms 117,653 input cells, 439,167 initial peaks, 434,240 peaks after
the detection filter, and exclusion of LSI1 after its depth correlation of 0.9427.
That run wrote its embedding arrays and metadata before failing in a subsequent
categorical-label plotting step. It must not be described as a fully completed
plotting run, or as a failure to calculate the embeddings.

Both embeddings were calculated using the 117,653-cell paired input cohort. The
111,347 plotted/model-cohort cells retain their saved coordinates. RNA coordinates
match at the stored float32 precision; the maximum difference when reading the
two decimal text exports as float64 is below 1e-6. No integration was rerun.

**Batch covariates differ:** RNA uses `source_GSE` (four studies); ATAC uses `batch`
(21 source-library groups). The ATAC batch field must not be described as four
source-study groups. Exact per-cell covariates and paired coordinates are in
`data/embeddings/final_cohort_paired_UMAP_coordinates.tsv.gz`.

Original source paths and requirements are preserved as an archival record;
configuration is required for reuse on another system. Original expression and
ATAC matrices were not processed anew during this source/coordinate verification.
