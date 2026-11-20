# Repository audit and validation status

## Existing material

The original checkout contained one Colab notebook, a README and an empty `eg`
file. The notebook already supplied stable WILDS sample IDs, disjoint split
checks, image-only ResNet-18 training, deterministic per-epoch dynamics, earliest
tie-breaking by overall validation accuracy, retrospective subgroup evaluation,
signal analysis and an optional JTT-style comparison.

The saved output records seed 42 completing with test overall accuracy 0.8676
and worst-group accuracy 0.4564, then seed 43 starting. It does not contain a
completed three-seed raw-result export or executed signal-analysis/JTT output.
The old README's approximately 86.9%/53.8% multi-seed means and 0.88 early-loss
ROC-AUC cannot be independently reconstructed from the checked-in artifacts.
They have not been declared false or silently replaced with new measurements:
the original text is preserved in `docs/ORIGINAL_README.md`, pending import of
the original Colab/Drive CSVs and checkpoints.

## Changes

- Move the notebook's prematurely placed signal-analysis cells after training;
  retain its saved historical outputs and identify them as historical.
- Replace the notebook's certificate-verification bypass with a verified-HTTPS
  fallback to the pinned Waterbirds URL published by WILDS.
- Add command-line training with the notebook's architecture/optimiser setup,
  fixed early selection rules, a random-selection control and epoch resume.
- Keep class labels and subgroup labels conceptually distinct throughout docs.
- Add atomic checkpoint/report writes, seed/environment/source/data provenance,
  strict completeness checks and per-seed/paired analysis with subgroup counts.
- Add synthetic and real-data pilots plus single-GPU Slurm scripts with one
  concurrent array task. No agent-side cluster access or training execution.

## Completed validation and results

The operator reported all 17 base tests passing on Windows and Stanage. Six
additional detector tests bring the source suite to 23 cases. The synthetic
pipeline validated four methods; deliberate interruption/resume reproduced the
uninterrupted model weights, epoch metrics and predictions for every method.
The agent has not rerun tests or training; source and report checks are separate.

Input preparation completed with 11,788 images and verified pretrained weights.
Two deployment failures happened before training: an unavailable Python module
and a missing custom environment variable. A module-based Python also raised
illegal instructions. The working GPU environment uses Anaconda3/2025.06-1,
Conda Python 3.12 and Torch 2.11.0+cu128 on an NVIDIA A100-SXM4-80GB. Scripts
now have explicit environment export and defaults for environment/data/weights.

The real-data two-epoch pilot completed successfully and validated four results.
The full campaign completed three seed tasks, training all four methods for
30 epochs. Operator-supplied accounting reported COMPLETED, exit 0:0 and
elapsed 00:17:41/00:17:43/00:17:45. Full analysis validated all 12 results.
This verifies synthetic checkpoint recovery, not an interrupted real GPU run.

The CPU detector follow-up improved top-10% precision and AP in all three model
seeds. Its class-relative score has not been used in a new retraining campaign.
The original raw-loss retraining did not outperform the random control on mean
worst-group accuracy. See RESULTS.md for exact findings and variability.

Downloaded raw data, configs, analysis outputs and source snapshots are in
evidence/stanage. Local data-only re-analysis reproduced all imported CSV
reports, and measured source hashes matched the archived training/detector
code. The image dataset and model checkpoints are excluded. The archive has
no git history, so the recorded git revision is null rather than fabricated.

GitHub Actions is prepared to run tests, synthetic recovery and recorded-data
validation after a future push; it has not been triggered by this local work.
Original notebook outputs remain historical and have not been rerun.
