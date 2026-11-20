# Waterbirds Training Dynamics

An undergraduate research project investigating whether a model's behaviour during training can reveal when it has learned an unreliable shortcut.

## Research question

> To what extent can early, label-free training dynamics identify training examples associated with shortcut-conflicting subgroups in Waterbirds, and can using these signals improve worst-group accuracy without subgroup labels?

## Project overview

The [Waterbirds](https://wilds.stanford.edu/datasets/#waterbirds) dataset contains images of landbirds and waterbirds shown against land or water backgrounds. Because bird type and background are correlated, an image classifier can perform well overall while relying too heavily on the background and failing on the less common bird/background combinations.

This project uses an image-only ERM baseline and records per-example training behaviour over time. Candidate signals include loss, confidence, correctness, learning time, and forgetting events. Subgroup metadata is reserved for retrospective evaluation; it is not used to train the model or select examples.

## Current findings

The baseline was trained for 30 epochs using seeds 42, 43, and 44.

- Mean test accuracy: approximately **86.9%**.
- Mean test worst-group accuracy: approximately **53.8%**.
- The weakest subgroup was consistently waterbirds on land backgrounds.
- Mean loss by epoch 5 identified shortcut-conflicting examples with approximately **0.88 ROC-AUC**.
- The top 1% ranked by mean loss contained approximately **39%** shortcut-conflicting examples, compared with a baseline prevalence of about **5%**.
- A preliminary JTT comparison was implemented. In this run, it did not consistently improve worst-group accuracy over ERM.

These are preliminary results from three seeds, not a claim that the method generalises to all datasets or settings.

## Notebook

The main reproducible artifact is:

- [`01_baseline_waterbirds.ipynb`](01_baseline_waterbirds.ipynb)

The notebook includes:

- WILDS Waterbirds dataset setup.
- ImageNet-pretrained ResNet-18 ERM training.
- 30-epoch runs for seeds 42, 43, and 44.
- Stable sample IDs based on original WILDS training indices.
- Shuffled training with augmentation.
- Deterministic training-dynamics evaluation after every epoch.
- Overall, subgroup, and worst-group accuracy evaluation.
- Label-free signal analysis using full and early training windows.
- An optional JTT comparison.

## Reproducibility

The notebook is designed for Google Colab with a single GPU. To run a fresh baseline, use:

```python
SEEDS = [42, 43, 44]
RUN_SEEDS = [42, 43, 44]
```

For more reliable Colab execution, use:

```python
"num_workers": 0
```

The training loop runs seeds sequentially. Each seed resets the Python, NumPy, and PyTorch random states and creates a fresh model, optimizer, DataLoader, and results container.

If previous outputs were saved to Google Drive, the notebook includes an optional restore cell. This allows the signal-analysis cells to be run without retraining the baseline.

## Output files

For each seed, the baseline produces:

```text
results/seed_42/training_dynamics.csv
results/seed_42/epoch_metrics.csv
results/seed_42/validation_predictions.csv
results/seed_42/test_predictions.csv
```

The same files are produced for seeds 43 and 44. Additional analysis outputs include:

```text
results/multiseed_baseline_summary.csv
results/seed_{42,43,44}/signal_scores.csv
results/seed_{42,43,44}/signal_scores_first_{5,10}_epochs.csv
results/multiseed_detection_metrics.csv
results/multiseed_early_detection_metrics.csv
results/signal_definitions.csv
```

Checkpoints are saved as:

```text
checkpoints/resnet18_waterbirds_erm_seed_42_30epochs.pt
checkpoints/resnet18_waterbirds_erm_seed_43_30epochs.pt
checkpoints/resnet18_waterbirds_erm_seed_44_30epochs.pt
```

## Interpretation and limitations

Overall accuracy alone hides the subgroup failure in this task, so worst-group and per-group accuracy are reported alongside it. The label-free signals are evaluated against subgroup metadata only after they have been calculated.

The current evidence supports the more cautious conclusion that early training dynamics can identify examples associated with shortcut-conflicting subgroups in Waterbirds. Further work is needed to establish whether using those signals improves robustness, how they compare with a carefully matched JTT implementation, and whether the findings transfer to a second dataset.

## Planned next steps

1. Validate the early-signal results separately for each seed.
2. Finalise a canonical JTT comparison.
3. Retrain using a pre-specified early mean-loss signal.
4. Compare ERM, JTT, and dynamics-based retraining on worst-group accuracy.
5. Repeat the completed pipeline on a second shortcut-learning dataset.

TRAK and other data-attribution methods are possible later extensions, not part of the current core experiment.

## Acknowledgements

This project uses the [WILDS benchmark](https://wilds.stanford.edu/) and the Waterbirds dataset for studying distribution shift and subgroup robustness.
