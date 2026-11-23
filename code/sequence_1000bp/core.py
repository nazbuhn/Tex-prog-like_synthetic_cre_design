"""Motif-free library objectives and evaluation; no known PWM resources."""
import numpy as np
from design_core import margin, robust, longest_run, dna
from io_helpers import need


def encode(sequence):
    need(set(sequence) <= set('ACGT'), 'Non-ACGT design')
    return np.array(['ACGT'.index(c) for c in sequence], np.uint8)


def objective(profiles, target, floor, penalty=10.):
    gap = robust(margin(profiles, target))
    deficit = np.maximum(np.log1p(floor) - np.log1p(profiles[..., target]), 0).mean(axis=-1)
    return gap - penalty * deficit


def best_step(profiles, target, floor):
    feasible = (profiles[..., target] >= floor - 1e-7).all(axis=-1)
    score = robust(margin(profiles, target))
    if feasible.any():
        return int(np.argmax(np.where(feasible, score, -np.inf))), True
    return int(np.argmax(objective(profiles, target, floor))), False


def qc(sequence):
    gc = float(np.mean((sequence == 1) | (sequence == 2)))
    return dict(GC_fraction=gc, longest_homopolymer=longest_run(sequence),
                sequence_QC_pass=bool(.2 <= gc <= .8 and longest_run(sequence) <= 8))


def ratio(profiles, target):
    off = np.delete(profiles, target, axis=-1).max(axis=-1)
    return profiles[..., target] / np.maximum(off, 1e-8)


def distinct(a, b, distance=15):
    return min(int((a != b).sum()), int((a != 3-b[::-1]).sum())) >= distance


def freeze_shortlist(records, sequences, maximum=128, parent_cap=4, distance=15):
    """Selection contexts + primary model only. Never receives final audit data."""
    ranked = sorted((r for r in records if r['selection_eligible']),
                    key=lambda r: (-r['selection_score'], r['design_id']))
    chosen = []; counts = {}
    # First give each natural parent / independent random start a chance.
    for cap in (1, parent_cap):
        for r in ranked:
            i = r['index']; parent = r['parent_id']
            if i in chosen or counts.get(parent, 0) >= cap:
                continue
            if not all(distinct(sequences[i], sequences[j], distance) for j in chosen):
                continue
            chosen.append(i); counts[parent] = counts.get(parent, 0) + 1
            if len(chosen) == maximum:
                return chosen
    return chosen


def sequence_key(sequence):
    return min(dna(sequence), dna(3-sequence[::-1]))
