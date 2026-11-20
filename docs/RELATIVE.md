# Class-relative retraining versus class-matched random selection

Status: the cluster campaign completed and its negative result is reported below.
The latest raw follow-up files are still on Stanage; prior evidence is retained.

## Completed result and evidence status

The operator completed the six new model runs and the analyser reported
`Validated 6 new method/seed results with exact class-matched selections`.
The rounded three-seed console summaries were:

| Method | Mean overall accuracy | Mean worst-group accuracy |
|---|---:|---:|
| Previous ERM reference | 87.50% | 43.98% |
| Class-relative retraining | 87.30% | 43.87% |
| Class-matched random | 87.94% | 49.69% |

The primary paired worst-group differences (class-relative minus control) were
-4.67, -5.45 and -7.32 percentage points for seeds 42/43/44. All were negative;
the mean difference was approximately -5.8 points. This experiment did not
demonstrate a robustness advantage from the improved detector. Matching class
counts removed one explanation for the original comparison, but scored
selection still lost to the random control under the fixed settings.

These are the operator's reported and validated cluster results. A transcript
is retained at [evidence/reported/relative-v1/terminal-output.txt](../evidence/reported/relative-v1/terminal-output.txt).
The raw `runs/relative-v1` files and `results/relative-v1` reports have **not yet
been downloaded into this repository**. Therefore this write-up does not claim
independent local re-analysis, exact unrounded means or measured significance
for this follow-up. The original comparison and CPU detector evidence remain
fully included and independently re-analysed in `evidence/stanage`.

The result supports the narrower conclusion that better retrospective subgroup
detection did not translate into better retraining with this weighting rule.
It does not establish that every class-relative strategy or JTT generally fails.
The original fixed rules below are retained so the negative result is reproducible.

## Question and fixed choices

Does selecting high class-relative early-loss images help worst-group accuracy
more than selecting random images with exactly the same bird-class counts?

The primary comparison is `class_relative minus class_matched_random`, measured
as the paired worst-group test-accuracy difference in percentage points for
each seed, followed by its three-seed mean and sample deviation. Overall accuracy
and all four group metrics are also reported. Both positive and negative seed
differences remain in the report; there is no automatic selection of a winner.

The main config keeps ResNet-18/ImageNet weights, 30 epochs, batch size 64,
AdamW lr=1e-4/weight_decay=1e-4, the original transforms, the first five ERM
epochs, a 10% selection budget and 10x upweighting. Seeds remain 42/43/44.
No learning-rate, weight, epoch-window, fraction or random-draw sweep is added.

The existing completed ERM histories choose the images; no original ERM or
other original method is retrained. Per seed:

1. Rank the saved five-epoch scores using the detector already implemented in
   `detect.py`, breaking ties by sample ID; take the top `ceil(0.1*N)`.
2. Count selected landbirds and waterbirds using training bird labels only.
3. Draw exactly those counts uniformly without replacement inside each class.
   One fixed control RNG seed is `model_seed + 10000`. The draw is from the full
   training class, so overlap with the scored selection is allowed and unbiased.
4. Train two fresh models from the same initial state as each other and the
   reference ERM. The initial model hash is checked before any optimiser update.

For the saved real-data histories the scored top 480 contained 368 landbirds
and 112 waterbirds for every seed. The code derives rather than hardcodes these
counts, then requires the control's exact counts to agree. Both methods use
the same WeightedRandomSampler, N draws/epoch, optimiser update budget and
selected/unselected weights. Their expected weighted class masses are equal.
Individual sampled class counts per minibatch need not be identical.

No backgrounds, subgroup labels, validation scores or test scores enter image
selection. Validation overall accuracy chooses the final checkpoint with
earliest ties; test/subgroup predictions are evaluated afterwards. The original
ERM is shown as a previously measured reference, not falsely described as a
fresh model in the new allocation. Strict data/split/weights/software/GPU and
initial-model checks keep that comparison traceable.

This is an exploratory continuation on an already examined dataset. The seeds
reuse the same images and the random control has one fixed draw per model seed.
Neither a positive result nor improved generalisation is guaranteed. Three
seeds do not justify significance claims. The earlier detector gain is not
relabeled as a downstream gain; the new experiment has its own raw outputs.

## Local Windows checks

Use the existing Waterbirds environment; no new dependencies are needed. There
are 35 tests after the 12 new selection/reference/report tests are added.

```powershell
cd D:\GitHub\waterbirds-training-dynamics
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Create a fresh tiny reference, then run the new synthetic comparison. These are
artificial-image plumbing checks, not Waterbirds research results:

```powershell
.\.venv\Scripts\python.exe train.py --config configs/smoke.json --seed 42 --synthetic --device cpu --output scratch/relative-reference
.\.venv\Scripts\python.exe train_relative.py --config configs/relative-smoke.json --reference-root scratch/relative-reference --seed 42 --synthetic --device cpu --output scratch/relative-full
.\.venv\Scripts\python.exe analyse_relative.py scratch/relative-full --reference-root scratch/relative-reference --config configs/relative-smoke.json --output scratch/relative-analysis --allow-synthetic
```

Expect two new method/seed results. The equal-budget synthetic reference also
appears in its table. Then test deliberate interruption/resume:

```powershell
.\.venv\Scripts\python.exe train_relative.py --config configs/relative-smoke.json --reference-root scratch/relative-reference --seed 42 --synthetic --device cpu --output scratch/relative-resumed --stop-after-epoch 1
.\.venv\Scripts\python.exe train_relative.py --config configs/relative-smoke.json --reference-root scratch/relative-reference --seed 42 --synthetic --device cpu --output scratch/relative-resumed --resume
.\.venv\Scripts\python.exe check_resume.py scratch/relative-full scratch/relative-resumed --config configs/relative-smoke.json
```

Both new methods should have identical final models, epoch metrics and
predictions across uninterrupted/resumed runs. Use fresh scratch output names
if repeating these checks; do not delete or overwrite previous scientific runs.

## Transfer to Stanage without pushing

Windows PowerShell, after the checks pass:

```powershell
tar -czf scratch/relative-update.tar.gz train.py check_resume.py train_relative.py relative_rules.py analyse_relative.py configs tests slurm docs
scp scratch/relative-update.tar.gz aca24gl@stanage.shef.ac.uk:~/
```

Stanage login terminal:

```bash
cd ~/waterbirds-training-dynamics
tar -xzf ~/relative-update.tar.gz
sed -i 's/\r$//' slurm/relative.sbatch slurm/relative-pilot.sbatch
```

This updates source/config/tests only. It does not change `runs/comparison-v1`,
data, weights or the Conda environment. Updating `train.py` adds explicit
selection/initialisation hooks; its ordinary ERM/JTT/early-loss/random behaviour
is unchanged. Old run resumption intentionally rejects changed source hashes;
use the original archived sources if an old campaign ever needs resumption.

## CPU preflight

Request an allocation if currently on a login node; do not nest allocations:

```bash
salloc --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=4G --time=00:10:00
```

Inside it:

```bash
cd ~/waterbirds-training-dynamics
module purge
module load Anaconda3/2025.06-1
source "$EBROOTANACONDA3/etc/profile.d/conda.sh"
conda activate "$PWD/.conda-a100"
python -m unittest discover -s tests -v
python relative_rules.py runs/comparison-v1 --config configs/relative.json --seed 42 --output scratch/relative-plan-42.json
exit
```

Expect 480 selected per method with 368 landbirds/112 waterbirds, and identical
expected class sampling masses. This only reads the saved histories; no GPU,
input download or training is used. A reference/config mismatch fails before
training rather than silently recomputing or changing the original protocol.

## Real-data pilot, then full GPU campaign

From the login node, submit only the pilot first:

```bash
cd ~/waterbirds-training-dynamics
bash -n slurm/relative-pilot.sbatch
sbatch --export=ALL --test-only slurm/relative-pilot.sbatch
sbatch --export=ALL slurm/relative-pilot.sbatch
```

The pilot reuses the full 30-epoch reference ERM and its five-epoch selection,
but trains the two new models for only two epochs. It has a 15-minute ceiling.
It is not compared against the 30-epoch ERM as a scientific result. Inspect:

```bash
sacct -j PILOT_JOB --format=JobID,State,ExitCode,Elapsed
tail -n 20 waterbirds-relative-pilot-PILOT_JOB.out
```

In an allocated CPU analysis session, with Conda activated:

```bash
python analyse_relative.py runs/relative-pilot-PILOT_JOB --reference-root runs/comparison-v1 --config configs/relative-pilot.json --output results/relative-pilot-PILOT_JOB
```

Expect two validated new method/seed results. Only after that passes, submit
the 30-epoch comparison from the login node:

```bash
bash -n slurm/relative.sbatch
sbatch --export=ALL --test-only slurm/relative.sbatch
sbatch --export=ALL slurm/relative.sbatch
squeue --me
```

The array runs seeds 42/43/44, at most one task at a time. Each requests one
A100, four CPUs, 16 GB and a 30-minute maximum. It trains two models per seed:
six fresh models total. The time request is a limit, not a new measured runtime.
Environment/data/weights default to the proven repository paths, and input
reference defaults to `runs/comparison-v1`. Overrides use `WB_CONDA_ENV`,
`WB_DATA_ROOT`, `WB_WEIGHTS`, `WB_REFERENCE_ROOT`, `WB_RELATIVE_OUTPUT` and
`WB_RELATIVE_PILOT_OUTPUT`; they do not reuse the old campaign's `WB_OUTPUT`.

Do not edit training sources or packages while the campaign is active: metadata
and checkpoint identity checks reject mixed sources/environments across seeds.
If a task is interrupted, confirm it ended, then resubmit only that index into
the same output, for example `sbatch --export=ALL --array=1 slurm/relative.sbatch`.
The last completed epoch is resumed. Never run two jobs for the same seed/output.

## Final analysis

In an allocated CPU session with Conda activated:

```bash
python analyse_relative.py runs/relative-v1 --reference-root runs/comparison-v1 --config configs/relative.json --output results/relative-v1
```

Expect six validated new method/seed results and three old ERM reference rows.
The report verifies actual selection IDs, exact class quotas, initialisation,
environment, checkpoint choice, data IDs and metrics, then exports raw per-seed
results, means/sample deviations, the paired primary control differences and
a figure. Neither partial campaigns nor missing controls are averaged silently.
Keep reference histories, selection plans, predictions and metadata with the
new reports; retain both positive and negative results before updating the README.
