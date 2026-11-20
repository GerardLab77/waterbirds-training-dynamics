# Detector follow-up: class-relative early loss

The completed baseline detected shortcut-conflicting images with per-seed
ROC-AUC 0.835–0.908. Its top 10% contained 23.3–32.7% conflicting images,
versus 5.0% dataset prevalence. Early-loss retraining did not beat the random
control on mean worst-group accuracy. These original findings are retained.

The next question is narrower: does comparing loss within each bird class
improve the detector's ranking over plain mean loss? No more model training is
needed: use the saved ERM histories from the completed comparison.

## Completed findings

The fixed follow-up has now run on CPUs using the saved histories. Mean top-10%
precision increased from 26.81% to 28.68% (+1.875 percentage points). Mean AP
increased from 0.27612 to 0.35702 (+29.3% relative), and mean AUC from 0.86598
to 0.87492. Precision and AP improved for every seed. The global top-480
selection found 112/157/117 conflicting examples with raw loss and 132/161/120
with class-relative loss. Within-class AUC and AP stayed unchanged, as expected.
Reports, raw data and source snapshots are included in evidence/stanage.
See [RESULTS.md](RESULTS.md) for the full interpretation and reproduction steps.
The subsequent class-matched retraining comparison was negative in all three seeds; see RELATIVE.md for the reported results and pending raw evidence.

## Fixed scoring rules

- Baseline: mean loss over the same first five epochs as the original campaign.
- Candidate: convert that mean loss to its percentile rank within its bird-type
  training class. Midranks keep tied losses tied; a singleton class scores 0.5.

The candidate uses the existing bird label, not background or subgroup labels.
Both scores are finalised before retrospective metadata enter evaluation.
The hypothesis is that raw losses may over-rank a globally harder class;
class-relative ranks compare images with others of the same bird type.
This is a measured improvement over the raw-loss baseline here, not a claimed novel algorithm.

The primary metric is precision among the top 10%, matching the existing
selection budget. Also report AP, ROC-AUC, recall, enrichment, selected class
counts and per-class performance. A per-class scope selects 10% of that class;
it is not a count of class members inside the globally selected set.
Within-class ordering is unchanged, so within-class AUC/AP should match the
baseline. The experiment isolates cross-class calibration, not new within-class
information. Random expected precision is the relevant scope's prevalence.

There is one fixed candidate, one early window and one selection budget.
Report both methods and all three seeds, whether results improve or worsen.
This is exploratory because the initial results have already been inspected.
Seeds reuse the same images; they are not independent held-out image sets.
Improved detector metrics would not establish improved downstream robustness.

## Operator commands

Copy `detect.py` and `tests/test_detection.py` to their matching paths on
Stanage. Existing training modules/config/checkpoints are not changed.

```bash
python -m unittest discover -s tests -v
python detect.py runs/comparison-v1 --config configs/comparison.json --output results/detection-v2
```

Run in an allocated CPU session with the working Conda environment active.
There are 23 tests after adding this module. No GPU, training, dataset download
or checkpoint loading is involved in this follow-up. The script first validates
the original 12 results, then exports scores, per-seed/per-class metrics, means
and sample deviations, paired changes, and protocol/source/input hashes.
It refuses an existing output directory so previous reports remain available.

Before another expensive training campaign, inspect these detector comparisons
and the existing selection label counts. If the candidate does not improve
the primary metric consistently, retain that result rather than claiming a
better detector or selecting a favourable secondary metric after seeing it.
