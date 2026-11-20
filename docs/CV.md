# CV wording based on completed evidence

- Developed class-relative training-loss scoring for Waterbirds, improving mean
  average precision from 0.276 to 0.357 (29.3% relative) and top-10% precision
  from 26.81% to 28.68% across three model seeds, without using subgroup labels
  to construct scores.
- Built a reproducible ResNet-18 comparison of four training approaches across
  three seeds on Sheffield's Stanage A100 GPUs, with Slurm job arrays,
  epoch-level checkpoints and verified synthetic interruption/resume behaviour.

The detector follow-up is exploratory and uses the same training images across
seeds. The subsequent class-matched retraining comparison was negative in all three seeds; the latest raw follow-up files remain on Stanage.
The earlier mean-loss retraining did not outperform the random-selection control
on mean worst-group accuracy. Never describe the 29.3% AP improvement as a
classifier-accuracy or general robustness improvement. See RESULTS.md for the
raw evidence, exact controls and variability.
