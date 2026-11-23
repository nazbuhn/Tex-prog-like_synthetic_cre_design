"""Derived export alias for native CREsted merged pattern dictionaries.

Native greedy post-hoc merging may return only pattern/classes/instances/ic,
omitting a redundant top-level ppm field. Derive that export field from the
UNCHANGED selected representative using CREsted's own PPM conversion.
No pattern is selected, removed, merged or reoriented by this adapter.
"""
import hashlib
import inspect
import numpy as np


def install(workflow, run):
    original = workflow.native_cluster

    def native_cluster_with_export_alias(c, files, cfg, mode):
        patterns, args = original(c, files, cfg, mode)
        from crested.tl.modisco._modisco_utils import _pattern_to_ppm
        rows = []
        for key, data in patterns.items():
            representative = data['pattern']
            # Use the exact representative chosen by native CREsted, including
            # native trimmed seqlets, A/C/G/T order and pseudocount convention.
            ppm = np.asarray(_pattern_to_ppm(representative), dtype=float)
            if ppm.ndim != 2 or ppm.shape[1] != 4 or len(ppm) == 0:
                raise RuntimeError('Unexpected native representative PPM shape: ' + str(key))
            if not np.isfinite(ppm).all() or np.any(ppm < 0) or not np.allclose(ppm.sum(1), 1):
                raise RuntimeError('Invalid native representative probabilities: ' + str(key))
            had_top_level_ppm = 'ppm' in data
            data['ppm'] = ppm
            rows.append(dict(cluster=str(key), representative_id=str(representative.get('id', '')),
                width=len(ppm), source_instances=len(data['instances']),
                source_classes=len(data['classes']), had_top_level_ppm=had_top_level_ppm,
                probabilities_from='native _pattern_to_ppm on the unchanged selected representative'))
        workflow.dump(run / ('PPM_EXPORT_COMPATIBILITY_' + mode + '.json'), dict(
            stage='export compatibility only', clustering_mode=mode, clusters=len(patterns),
            conversion_source_file=inspect.getsourcefile(_pattern_to_ppm),
            conversion_source_sha256=hashlib.sha256(inspect.getsource(_pattern_to_ppm).encode()).hexdigest(),
            native_clustering_unchanged=True, motifs_filtered_by_adapter=0,
            representatives_changed_by_adapter=0, records=rows))
        return patterns, args

    workflow.native_cluster = native_cluster_with_export_alias
