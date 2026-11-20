# Completed Waterbirds experiments

## Evidence and verification

The operator completed three A100 array tasks, one model seed per task, using
Slurm on Stanage. Each trained ERM, early-error JTT-style, early-mean-loss and
random-selection models for 30 epochs. There are 12 completed method/seed
results. The original config and source snapshots accompany the raw results in
[`evidence/stanage`](../evidence/stanage). No model checkpoints or dataset images
are included in the evidence archive.

Local re-analysis validates all 12 results, fixed selections, identical model
initialisation per seed, checkpoint histories, split IDs and subgroup metrics.
Source hashes for all three model seeds match the archived `train.py` and
`waterbirds_core.py`. The detector source matches its recorded protocol hash.
Reports are recomputed from saved data without new training or test execution.
The absence of `.git` in the transferred source caused the recorded git revision
to be null; source hashes and archived files provide the code provenance.

Dataset split sizes are 4,795 training, 1,199 validation and 5,794 test images.
The three model seeds use the same images, not independent held-out datasets.
Checkpoint selection uses overall validation accuracy; all subgroup metrics are
retrospective. Overall accuracy is raw fraction correct, not WILDS' weighted
Waterbirds aggregate. See [PROTOCOL.md](PROTOCOL.md) for the exact rules.

## Original matched retraining

Mean and sample standard deviation over seeds 42/43/44, in percentage points:

| Method | Overall test accuracy | Worst-group test accuracy |
|---|---:|---:|
| ERM | 87.50 ± 1.01 | 43.98 ± 1.01 |
| JTT-style | 87.33 ± 0.10 | 39.46 ± 5.19 |
| Early mean loss | 86.67 ± 0.83 | 47.40 ± 7.24 |
| Random selection | 88.42 ± 0.24 | 48.60 ± 2.97 |

Early mean-loss selection improved worst-group accuracy over ERM in two of
three seeds, but did not beat random selection on the mean. The JTT-style
configuration was worse than ERM for all three seeds; it is a matched variant,
not a reproduction or general refutation of the original JTT paper.

The mean early-loss gain over ERM is 3.43 percentage points using full-precision
saved values (the difference between rounded displayed means is 3.42). Random
selection is 4.62 points above ERM and 1.19 points above early-loss selection.
Report per-seed differences and variability, rather than a significance claim.

Reports: [summary](../evidence/stanage/results/comparison-v1/summary.csv),
[per-seed metrics](../evidence/stanage/results/comparison-v1/per_seed.csv),
[paired differences](../evidence/stanage/results/comparison-v1/paired_differences.csv),
[comparison figure](../evidence/stanage/results/comparison-v1/comparison.png).

## Detector follow-up

The plain detector averages loss over ERM epochs 1–5. The follow-up converts
that score to a percentile within the training bird-type class. Background
labels do not enter scoring. The same five-epoch window and top-10% budget
are retained; both detectors are evaluated on the same saved histories.

| Metric, mean over model seeds | Mean loss | Class-relative loss |
|---|---:|---:|
| Precision among top 10% (primary) | 26.81% | 28.68% |
| Average precision | 0.2761 | 0.3570 |
| ROC-AUC | 0.8660 | 0.8749 |
| Recall among top 10% | 53.61% | 57.36% |
| Enrichment over 5.005% prevalence | 5.36x | 5.73x |

Top-10% precision improved by 1.875 percentage points, or 7.0% relative.
Average precision improved by 0.08090, or 29.3% relative. These are different
metrics; 29.3% is not an accuracy or downstream robustness improvement.
Both metrics improved for all three seeds. The top-10% budget selects 480 images:

| Seed | Conflicting images found, raw | Found, class-relative | Precision change |
|---|---:|---:|---:|
| 42 | 112 | 132 | +4.17 pp |
| 43 | 157 | 161 | +0.83 pp |
| 44 | 117 | 120 | +0.63 pp |

### What changed in the ranking

There are 3,682 landbirds and 1,113 waterbirds in training. Waterbirds are only
23.2% of the examples but occupy 39.0–65.4% of the raw-loss top-480 selections.
Class-relative ranks put 112 waterbirds (23.3%) in the global selection for each
seed. This shifts the selection towards the training class proportions.
Within-class AUC and AP stay exactly unchanged, as expected: the method changes
cross-class calibration, not within-class ordering or the model's information.
The archived per-class metrics verify this distinction.

This follow-up was proposed after inspecting the original experiment. It is
exploratory evidence on the same training images, not a blinded held-out study
or proof of generalisation. A subsequent class-relative retraining comparison has now completed: the
operator reported that it lost to an exact class-matched random control in all
three seeds. The latest raw files are not yet imported; see RELATIVE.md and the
reported transcript. Do not attribute the detector score improvement to better
classifier robustness.

Reports: [summary](../evidence/stanage/results/detection-v2/summary.csv),
[all seed/class metrics](../evidence/stanage/results/detection-v2/metrics.csv),
[paired changes](../evidence/stanage/results/detection-v2/paired_differences.csv),
[protocol and input hashes](../evidence/stanage/results/detection-v2/protocol.json).

## Class-matched retraining follow-up: negative result

The operator's cluster analysis validated six new model/seed results. Mean
worst-group accuracy was 43.87% for class-relative retraining and 49.69% for
class-matched random selection. The scored selection lost by 4.67, 5.45 and
7.32 percentage points across the three seeds (about 5.8 points on average).
Both methods used exact matched bird-class counts and the frozen protocol.

The console summary is transcribed in
[`evidence/reported/relative-v1`](../evidence/reported/relative-v1/terminal-output.txt).
Full new histories, predictions and raw CSV reports remain on Stanage. These
reported findings have not yet been independently recomputed locally. The
earlier original/detector raw evidence and reproduction commands below are
unchanged. See [RELATIVE.md](RELATIVE.md) for the complete follow-up protocol
and evidence limitation. No new detector generalisation or robustness gain is
claimed from this negative experiment.

## Reproduce reports from the included evidence

In an environment with the pinned packages, from the repository root:

```bash
python analyse.py evidence/stanage/runs/comparison-v1 --config evidence/stanage/configs/comparison.json --output scratch/recheck-comparison
python detect.py evidence/stanage/runs/comparison-v1 --config evidence/stanage/configs/comparison.json --output scratch/recheck-detection
python scripts/plot_detection.py evidence/stanage/results/detection-v2/metrics.csv --output figures/detection-comparison.png
```

Use fresh scratch output names for repeat detector re-analysis. These commands
do not train models, download inputs or require a GPU.
