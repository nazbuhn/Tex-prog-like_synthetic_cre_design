# Recorded RNA preprocessing and export source

These seven original scripts were recovered from the CD8 exhaustion annotation
archive. Each matches the code checksum in its original run manifest. The final
export policy and all 46 source Step-04/Step-10 library-manifest links were
verified. All 111,347 cells in the final analysis cohort occur in the archived
161,773-cell final RNA export. The cohort-specific QC table retains their
original RNA QC, doublet-call and cycling-override fields.

Stages are count preparation/gene-symbol handling, initial QC review, Scrublet,
per-library cluster review, count export, DoubletDetection, and final RNA export.
The 08_cluster_review script is a per-library review stage, **not** the final
cross-study Harmony integration used for the manuscript embeddings.

Original source, final selection policy, reference mapping inputs, recorded
versions, manifests and summaries are included. No source was silently edited;
original paths require configuration before reuse. Personal RUN_SAVED launchers,
upload instructions, local environments, intermediate matrices and pilot local
annotation scripts are omitted. Historical branches can contain libraries not
included in the final four-study cohort; the deposited final-cohort labels and
QC table identify the actual cells used.

The selection policy includes historical sample-description/filename conflicts.
Those original flags are preserved, not presented as new condition assignments
or assumed resolved by this recovery. No QC, doublet model or annotation was
rerun during this verification.

The actual cross-study integration source and coordinate checks are now supplied
in `code/integration/`; Step 08 above remains a distinct per-library review.
