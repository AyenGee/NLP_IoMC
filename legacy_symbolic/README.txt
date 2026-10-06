LEGACY: the symbolic pilot (tag v2-symbolic-pilot)
==================================================
This folder is a frozen copy of the first implementation, an ADAPTATION of the Singh et al.
(2024) task (integer symbols, 4-way 3-shot, MLPs, learned positions, AdamW + cosine). It is kept
so the pilot results can still be regenerated and cited. It is not the main implementation.
The exact-task rebuild lives in the repository's src/, configs/, experiments/ and tests/.

The pilot RESULTS stay where they were written: results/collapse/, results/analysis/,
results/retrain/, results/tuning/, results/logs/, results/checkpoints/ and cluster_run/ (none
moved or deleted). To regenerate the pilot analysis (numbers + figures):
    python experiments/analyze_results.py --pilot
(from the repository root; it reads results/ and writes results/analysis_summary.json and
abstract/figures/*). To run pilot code itself, work inside this folder: its scripts locate
their own src/ and configs/ relative to their file.
