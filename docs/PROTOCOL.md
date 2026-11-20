# Matched retraining experiment

## Question

Can high early training loss identify shortcut-conflicting Waterbirds examples,
and does upsampling those examples improve worst-group accuracy over ERM,
early-error JTT-style retraining and a matched random-selection control?

This is a single-dataset, three-seed study. It is not a new QEC experiment,
causal proof of background reliance or a reproduction of the JTT paper's
reported accuracy. Seeds are replications of optimisation randomness on the
same dataset, not independent datasets.

## Fixed main configuration

`configs/comparison.json` specifies seeds 42/43/44, ImageNet-pretrained ResNet-18,
30 epochs per model, batch size 64, AdamW learning rate 1e-4 and weight decay
1e-4. Images are resized to 224x224 and ImageNet-normalised. Only training
uses random horizontal flips. Dynamics use deterministic evaluation transforms.
The rules and hyperparameters are fixed before the new comparison is run:

| Method | Selection | Second-stage behaviour |
|---|---|---|
| ERM | None | Ordinary shuffled training; also supplies the early dynamics |
| JTT-style | Examples misclassified after epoch 5 of ERM | Fresh model, selected images sampled with weight 10 |
| Early mean loss | Top 10% by mean loss across ERM epochs 1–5 | Fresh model, selected images sampled with weight 10 |
| Random control | Uniform seeded sample of the same size as early-loss selection | Fresh model, selected images sampled with weight 10 |

The first-stage ERM finishes 30 epochs so it also supplies a matched ERM result;
selection only reads its first five epochs. It does not read the best later
checkpoint's mistakes. Loss ties use ascending stable sample ID. There is no
subgroup-based threshold, class balancing, fraction sweep or test-based tuning.
An empty JTT error set falls back to ordinary shuffled ERM sampling.

WeightedRandomSampler draws exactly N training images with replacement per epoch
for reweighted methods. Selected/unselected sampling probability is proportional
to 10/1. ERM shuffles without replacement. Therefore all methods have the same
number of optimiser updates, but reweighted methods need not see every image in
every epoch. This is matched-compute error upsampling, not an exact recreation
of the JTT authors' architecture, optimiser, expanded dataset or tuning grid.
The random control matches the loss method's selection count, not JTT's variable
error count. Stage-one training is shared; total pipeline compute exceeds that
of a single ERM model and should not be hidden when describing method cost.

See the [JTT authors' code](https://github.com/anniesch/jtt) for the original
method and experiments. No upstream code is copied into the training pipeline.

## Information boundaries and metrics

Training loaders yield image, bird label and stable original WILDS index;
background metadata are dropped. The model sees only the image. Dynamics CSVs
contain IDs, seed, epoch, bird label, prediction, confidence, loss and correctness,
but no backgrounds or group IDs. Selection whitelists ID/loss/correctness fields.
These signals are **subgroup-label-free**, not class-label-free: cross-entropy
and correctness require the bird-type training label.

All methods reset RNGs and start with the same model-state hash for a given seed.
The checkpoint with highest raw overall validation accuracy wins; ties keep the
earliest epoch. No validation subgroup accuracy or test metric is used in this
decision. Test predictions are generated after checkpoint selection. Each method
performs its own validation selection under the same rule and epoch budget.

Group ID is `2 * background + bird_label`, matching the notebook's
`[background, y]` ordering. Reports include all four group counts and accuracies;
missing groups cause failure rather than an artificially high worst-group score.
Overall accuracy is the raw fraction correct, not WILDS' training-prior-weighted
Waterbirds aggregate. Report both raw overall and worst-group accuracy explicitly.

Training backgrounds are exported only after all four methods finish. Retrospective
analysis joins them to early scores to calculate per-seed ROC-AUC, average
precision, prevalence and top-10% precision. Tied scores get tie-aware AUC/AP.
The analysis does not pool repeated observations of the same images across seeds
into an enlarged independent sample. Three-seed summaries show mean and sample
standard deviation, plus every seed and paired changes from ERM in percentage
points. No significance claims or population confidence intervals are supplied.

The earlier notebook/README already exposed exploratory Waterbirds test and
subgroup results. This follow-up therefore is not a blind new held-out dataset
study. It prevents programmatic test/group-based selection in the new runs, but
cannot undo prior human exploration of this dataset. A second dataset would be
needed for a stronger generalisation claim later.

## Reproducibility and scope

The main script disables nondeterministic algorithms rather than silently
accepting them. It records config, seeds, package versions, source hashes,
pretrained-weight hash, metadata/split-ID hashes, GPU model and Slurm context.
The pinned Torch/vision versions match the versions printed in the saved
notebook; other packages are pinned for the new environment. This does not
promise bitwise equality to the earlier Colab results or across GPU platforms.

`last.pt` is an epoch checkpoint containing model/optimiser state, best model,
training history, ERM dynamics, and Python/NumPy/Torch/CUDA/loader RNG states.
Writes use atomic replacement. Resume rejects changed sources/config/data
identity/software/GPU model and never overwrites an unrelated run. An interrupted
epoch restarts from the prior completed epoch; it does not resume mid-minibatch.
Never run two jobs writing the same seed/output directory concurrently. Checkpoint
loading with `weights_only=False` is only for this project's own trusted files.

The synthetic configuration uses tiny deterministic images and a linear model.
It tests pipeline/checkpoint behaviour, not ResNet-18 or scientific performance.
The real-data pilot is required to exercise the actual model, data loader and GPU.
Dataset images, downloaded weights and large checkpoints are not committed.
Waterbirds' upstream terms restrict the dataset to non-commercial research and
education; the dataset is not redistributed here.
