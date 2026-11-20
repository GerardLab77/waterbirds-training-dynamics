# Running the comparison

The agent prepares code; you run tests and training. Start with the small checks.
Nothing in this guide requires another QEC campaign. Use a separate Waterbirds
environment and keep the run outputs/checkpoints on durable storage.

## Included completed evidence

The full GPU comparison and CPU detector follow-up are complete. Raw histories,
predictions, selections, metadata, reports and source snapshots are under
`evidence/stanage`. See [RESULTS.md](RESULTS.md) to reproduce those reports
without training, downloading data or requesting a GPU. The commands below
are for a fresh experiment and retain their own output roots.

## Windows: unit tests without ML packages

In local PowerShell, not in your Stanage SSH terminal:

```powershell
cd D:\GitHub\waterbirds-training-dynamics
python -m unittest discover -s tests -v
```

There are 23 data-rule, detector and analysis tests. They require only Python 3.10+.
They do not download data or run training.

## Local CPU synthetic pipeline and resume check

Use Python 3.12. This explicitly runs tiny synthetic training jobs. Do not use
the QEC environment. Install the CPU Torch build first, then the pinned packages:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe train.py --config configs/smoke.json --seed 42 --synthetic --device cpu --output scratch/full
.\.venv\Scripts\python.exe analyse.py scratch/full --config configs/smoke.json --output scratch/analysis --allow-synthetic
.\.venv\Scripts\python.exe train.py --config configs/smoke.json --seed 42 --synthetic --device cpu --output scratch/resumed --stop-after-epoch 1
.\.venv\Scripts\python.exe train.py --config configs/smoke.json --seed 42 --synthetic --device cpu --output scratch/resumed --resume
.\.venv\Scripts\python.exe check_resume.py scratch/full scratch/resumed
```

The analyser should validate four method/seed results. `check_resume.py` should
report identical model weights, epoch metrics and predictions for all four
methods. Use fresh output names if repeating uninterrupted runs. Synthetic
accuracy is meaningless and must never appear in the scientific result table.

## Stanage environment and input preparation

The original module-based Python environment failed on A100 nodes: Python
3.12.3 was absent, and the available Python 3.11.5 build raised an illegal
instruction before the project ran. The operator verified the common
Anaconda3/2025.06-1 module instead. A Conda Python 3.12 environment with
Torch 2.11.0+cu128 successfully detected an NVIDIA A100-SXM4-80GB.
This confirms environment/GPU visibility, not completed training.

Follow Sheffield's [Conda guide](https://docs.hpc.shef.ac.uk/en/latest/stanage/software/apps/conda.html)
and [GPU guidance](https://docs.hpc.shef.ac.uk/en/latest/stanage/GPUComputingStanage.html).
Do not run `conda init`, modify shared base, or mix Python modules with Conda.
Use an allocated CPU session for routine installs/downloads where possible;
only request a short GPU debugging allocation to verify GPU compatibility.
The common Conda module avoids dependence on a CPU-specific Python module.

In an allocated session, from the transferred checkout:

```bash
cd ~/waterbirds-training-dynamics
module purge
module load Anaconda3/2025.06-1
source "$EBROOTANACONDA3/etc/profile.d/conda.sh"
conda create --override-channels --channel conda-forge --prefix "$PWD/.conda-a100" python=3.12 pip -y
conda activate "$PWD/.conda-a100"
python -m pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python prepare_waterbirds.py --data-root data --weights-dir weights
```

If `.conda-a100` is already prepared, activate it rather than creating it again.
The existing images and weights are reused. Input preparation uses verified
HTTPS, a pinned WILDS archive URL and a pretrained-weight checksum; it does not
train. Keep environment/checkpoints on durable storage and outside Git.
The operator has already prepared 11,788 images and verified the weights.

For a short GPU compatibility check within a GPU allocation:

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

Official Torch binary wheels include CUDA runtime libraries; the operator's
A100 driver accepted this build. Do not add unrelated CUDA/cuDNN modules.
Release an interactive allocation with `exit` before returning to submission.

## Batch environment defaults

Both scripts explicitly request `--export=ALL` and default to `.conda-a100`,
`data/` and `weights/resnet18-f37072fd.pth` beneath the submission directory.
Exported `WB_CONDA_ENV`, `WB_DATA_ROOT` and `WB_WEIGHTS` override these defaults.
The comparison output defaults to `runs/comparison-v1`; use `WB_OUTPUT` to choose
a new campaign. Always submit from the repository root. This avoids requiring
custom shell variables merely to launch the prepared setup.

## Real-data GPU pilot

Back on the login node, from the repository root:

```bash
export WB_CONDA_ENV="$PWD/.conda-a100"
export WB_DATA_ROOT="$PWD/data"
export WB_WEIGHTS="$PWD/weights/resnet18-f37072fd.pth"
bash -n slurm/pilot.sbatch
sbatch --export=ALL --test-only slurm/pilot.sbatch
sbatch --export=ALL slurm/pilot.sbatch
squeue --me
```

The pilot requests one A100, four CPUs, 16 GB host RAM and at most 30 minutes.
It runs one seed and all four methods for two epochs, on the full real splits.
It validates the real pipeline, not the final five-epoch selection hypothesis.
The log ends with its actual `Completed comparison:` path. Check `sacct` exit
state and run the analyser on an allocated CPU session:

```bash
python analyse.py runs/pilot-JOB_ID --config configs/pilot.json --output scratch/pilot-analysis
```

Do not use the synthetic configuration as a GPU benchmark. GPU interactive
sessions are discouraged by Sheffield except for short essential debugging.

## Main campaign, only after pilot validation

From the login node, with the same exported WB variables:

```bash
export WB_OUTPUT="$PWD/runs/comparison-v1"
bash -n slurm/comparison.sbatch
sbatch --export=ALL --test-only slurm/comparison.sbatch
sbatch --export=ALL slurm/comparison.sbatch
squeue --me
```

The array has three tasks (seeds 42/43/44), capped at one concurrent task.
Each task trains four models on one A100 with four CPUs and 16 GB, with a
six-hour maximum per seed. This is a ceiling, not a runtime prediction; adjust
it using the real pilot and permitted limits. No multi-GPU training is used.
Do not submit concurrent jobs writing the same seed/output. Keep output roots
separate for scientifically different configs. Job scripts do not download data.

After an interrupted task, confirm the old task has ended, then resubmit only
its index into the same output root. For example, seed 43 is task index 1:

```bash
sbatch --array=1 slurm/comparison.sbatch
```

The script uses `--resume`: finished methods are retained and an unfinished
method resumes after its last atomically saved epoch. Changed config/source/
software/data/GPU identities are rejected. To change the experiment, choose a
new campaign output instead of trying to override those checks.

## Analysis and publication

In an allocated CPU session with the Conda environment activated:

```bash
python analyse.py runs/comparison-v1 --config configs/comparison.json --output results/comparison-v1
```

Expect 12 method/seed results. Analysis requires every seed and method complete,
verifies selection rules/checkpoint history/group metrics and checks matching
evaluation IDs. It exports per-seed results, three-seed means/sample deviations,
paired percentage-point differences, early detection metrics and a comparison
plot. No partial campaign is silently averaged.

Before publication, preserve configs, metadata, per-epoch metrics, ERM dynamics,
selection IDs, predictions and completion markers from `runs/` alongside the
analysis. They are ignored while jobs are active; copy small evidence files to
an intentional `evidence/` directory after validation. Keep large `.pt` files
outside Git. Import the original Colab CSVs separately if available; never mix
their checkpoint rules/environments into the new matched comparison's means.
