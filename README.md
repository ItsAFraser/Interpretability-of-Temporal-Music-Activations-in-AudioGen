# Interpretability-of-Temporal-Music-Activations-in-AudioGen

# Overview:

in recent works, researchers found neural networks trained on music appear to have learned implicit theories of musical structure through statistical learning alone[1]. In this paper feature activations are averaged over entire tracks which collapses the time dimension. However, music fundamentally changes through time. We propose analyzing feature activations over time to reveal structural and narrative features that track-level averaging cannot show us.

### **The Problem:**

the problem with the recent papers on Interpretable concepts in Large generative music models [1], is that the authors computed the mean activation across all time steps. Because the authors computed the average across all time steps, we lose information about when the activations occur in songs.

### **The Solution:**

instead of collapsing activations to a single scaler, instead we keep the activation time series for each feature and analyze its shape. for instance, different musical phenomena would have different characteristic temporal activation profiles:

**Static Properties:** Static properties are features that should remain active across a song. this includes:

- The Genre, Instrumentation, etc.

This type of profile is what is captured already in previous papers [1].

**Structural Markers:** structural markers are activations that activate to describe the ‘structure’ of a song, this may include:

- the intro, verse, chorus, bridge, etc.

these features should activate periodically and predictably based on where in the song we are listening.

**Narrative Arc:**  The Narrative Arc are features that activate when we ramp up, spike, or decay a narrative of a song. this includes:

- building tension, climatic release, fading out, etc.

**Local Events:** Local events are events that are brief and usually only occurs once. This includes:

- Drum fills, key changes, guitar solos, etc.

### The Process

**Dataset:** MTG-Jamendo

we need a dataset to train a SAE on longer audio files, otherwise the learned features may no capture those longer range patterns such as structural markers. not just 10 second clips [1]. we also need a dataset to evaluate and label features.

### CHPC Download Workflow (DTN + VAST)

For CHPC-compliant large transfers, use the dedicated downloader on a CHPC Data Transfer Node (`dtn05` to `dtn08`) and write to VAST scratch.

```bash
Scripts/SetupScripts/chpc_download_mtg_jamendo.sh \
    --dataset raw_30s \
    --type audio \
    --source mtg-fast \
    --target-dir /scratch/general/vast/u1406806/mtg-jamendo \
    --yes
```

Detailed instructions are in `Docs/chpc_mtg_jamendo_download.md`.

**Option A: Reuse Original SAE**

The plan is to reuse the original SAE [1], run the SAE over longer audio files, then analyze the temporal activation since the features learned on 10-second audio clips may still fire meaningfully on longer audio files. This helps reduce the cost from retraining a SAE.

**Option B: Train New SAE**

Train new SAEs on longer audio files from full length tracks this gives the SAE the opportunity to learn long range features. This is far more expensive, but could reveal new features that Nikhil Singh, et al. [1] couldn’t discover.

**Option C: Sliding window:**

instead of re-training an SAE, evaluate small chunks of the same track on the original authors SAE then stitching together the activations resulting from it. this means we wont have to fine tune or re train a SAE.

**Activation Extraction:**

for activation extraction we can follow the same process that Nikhil Singh did [1] except we do not average. for each track we can extract the full residual stream activation time series from MusicGens transformer layer.

then we can take the retrained (or new SAE) and encode each activation  vector to get the sparse feature activations for every time step. this will result in a matrix for each track [num_features × num_timesteps] rather than a single vector.

### Current Workflow in This Repo

1. Extract MusicGen decoder residual features from audio into `Data/Models/features`.
2. Train the SAE on timestep vectors with `--sample_mode frames` so time is not averaged away before training.
3. Re-run the trained SAE over full per-track feature sequences to obtain sparse activations for temporal analysis.

Example extraction command:

```bash
Scripts/TrainingScripts/run_extract_musicgen_features.sh \
    /Volumes/SSD_ALF/DataSets/MTG-Jamendo/Data/raw/raw_30s_audio_low \
    Data/Models/features \
    --max_files 16 \
    --max_duration_sec 30 \
    --decoder_layers 0,8,16,-1 \
    --metadata_json
```

This command writes separate outputs per layer under:

- `Data/Models/features/layer_00/...`
- `Data/Models/features/layer_08/...`
- `Data/Models/features/layer_16/...`
- `Data/Models/features/layer_final/...`

Example SAE training command without temporal averaging:

```bash
Scripts/TrainingScripts/run_train_sae_apple_quick.sh \
    Data/Models/features \
    Output/sae-quick-smoke \
    --sample_mode frames \
    --frame_stride 4 \
    --max_frames 4096 \
    --val_split 0.1
```

For larger CUDA/HPC runs, use:

```bash
Scripts/TrainingScripts/run_train_sae_hpc.sh \
    Data/Models/features/layer_final \
    Output/sae-hpc-layer-final \
    --sample_mode frames \
    --frame_stride 2 \
    --max_frames 0
```

Each training run now writes:

- `training_metrics.csv` with train and validation reconstruction/sparsity metrics per epoch.
- `sae_best.pt` selected by lowest validation loss (or train loss if validation is disabled).
- `run_manifest.json` with full run configuration for reproducibility.

### CHPC Queue-Friendly SLURM Workflow

The CHPC scripts are tuned for smaller, more schedulable GPU jobs by default.

Current defaults target the live Notchpeak social-science GPU allocation:

- account: `soc-gpu-np`
- partition: `soc-gpu-np`
- resources: `1 GPU`, `8 CPU`, `16 GB RAM`, `6 hours`

Before you submit any CHPC extraction or training jobs, make sure the CUDA environment exists. For CHPC, prefer the lean runtime env and place the environment itself on VAST scratch instead of home:

```bash
bash Scripts/TrainingScripts/rebuild_chpc_cuda_env.sh
export TEMPORAL_MUSIC_ACTIVATIONS_ENV_PREFIX=/scratch/general/vast/$USER/conda-envs/temporal-music-activations-cuda
python "$TEMPORAL_MUSIC_ACTIVATIONS_ENV_PREFIX/bin/python" -c "import torch, transformers; print(torch.__version__, torch.version.cuda, torch.backends.cuda.is_built(), torch.cuda.is_available())"
```

The CUDA environment must report a non-`None` CUDA version, `True` for `torch.backends.cuda.is_built()`, and `True` for `torch.cuda.is_available()` on a GPU node. If it prints a CPU-only build, remove and recreate the environment before submitting jobs.

The full [Temporal-Music-Activations-cuda.yaml](Temporal-Music-Activations-cuda.yaml) file is still useful for local notebook work. For CHPC batch jobs, use the lean [Temporal-Music-Activations-cuda-chpc.yaml](Temporal-Music-Activations-cuda-chpc.yaml) profile or the rebuild helper above.

If the checker fails before torch diagnostics, or `conda run -n temporal-music-activations-cuda python -c "import torch"` fails, rebuild the CHPC CUDA environment with the repo helper:

```bash
bash Scripts/TrainingScripts/rebuild_chpc_cuda_env.sh
```

This rebuild script does three things differently from a one-shot `conda env create`:

- installs the GPU-enabled PyTorch packages first with `--strict-channel-priority`
- installs only the runtime Conda stack needed for extraction and training
- places the environment, Conda cache, and pip cache on VAST scratch by default so home storage does not fill up again

On a CHPC GPU node, you can now run a single verification command instead of checking this manually:

```bash
bash Scripts/TrainingScripts/check_chpc_cuda_env.sh
```

This script loads the CHPC CUDA module, resolves the repo Python environment through `conda_utils.sh`, prints the detected GPU via `nvidia-smi`, and exits non-zero if PyTorch is CPU-only or cannot see the allocated GPU.

If you do not want to wait inside an interactive `srun` session, submit the short batch checker instead:

```bash
sbatch Scripts/TrainingScripts/chpc_check_cuda_env.slurm
```

Then inspect the resulting `CheckCudaEnv_<jobid>.out` file in the directory where you ran `sbatch`.

The checker wrapper now prefers a direct environment Python when it finds one at either `~/.conda/envs/temporal-music-activations-cuda/bin/python` or `/scratch/general/vast/$USER/conda-envs/temporal-music-activations-cuda/bin/python`, which avoids slow Conda activation during the verification job.

The CHPC scripts now fail fast if Python or the Conda environment is missing. If your Conda install is not in a standard location, set one of these overrides before `sbatch`:

- `TEMPORAL_MUSIC_ACTIVATIONS_PYTHON=/path/to/env/bin/python`
- `TEMPORAL_MUSIC_ACTIVATIONS_CONDA_SH=/path/to/etc/profile.d/conda.sh`
- `TEMPORAL_MUSIC_ACTIVATIONS_ENV_NAME=temporal-music-activations-cuda`
- `TEMPORAL_MUSIC_ACTIVATIONS_ENV_PREFIX=/scratch/general/vast/$USER/conda-envs/temporal-music-activations-cuda`

The CHPC Slurm entrypoints now move heavyweight caches off home storage by default:

- `HF_HOME=/scratch/general/vast/$USER/hf-cache`
- `TORCH_HOME=/scratch/general/vast/$USER/torch-cache`
- `TMPDIR=/scratch/general/vast/$USER/tmp`
- `XDG_CACHE_HOME=/scratch/general/vast/$USER/xdg-cache`

You can override any of these before `sbatch` if you want a different scratch layout.

The CHPC Slurm wrappers for environment checking, extraction, and training now prefer a direct environment Python when they find one. That avoids slow Conda activation during batch startup and makes the jobs more reliable on shared storage.

Feature extraction on CHPC now supports deterministic file slicing, which makes SLURM arrays straightforward. Each job can extract all MusicGen decoder layers for a chunk of files into the shared VAST feature tree.

Submit one pilot extraction chunk:

```bash
EXTRACT_CHUNK_SIZE=256 \
sbatch Scripts/TrainingScripts/chpc_extract.slurm
```

Submit several extraction chunks as a job array:

```bash
EXTRACT_CHUNK_SIZE=256 \
sbatch --array=0-7 Scripts/TrainingScripts/chpc_extract.slurm
```

Or use the helper script for a decent first run over all layers:

```bash
bash Scripts/TrainingScripts/submit_chunked_extract.sh
```

That defaults to 1,024 files split into 8 jobs of 128 files each. Override with `TOTAL_FILES` and `CHUNK_SIZE` when needed.

Useful extraction overrides:

- `RAW_AUDIO_DIR` defaults to `/scratch/general/vast/$USER/mtg-jamendo/raw_30s/audio`
- `SCRATCH_FEATURES_DIR` defaults to `/scratch/general/vast/$USER/sae_output/features`
- `EXTRACT_LAYERS` defaults to `0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,-1`
- `EXTRACT_CHUNK_SIZE` defaults to `256`
- `EXTRACT_FILE_START` can be set manually for one-off reruns
- `SBATCH_ACCOUNT` and `SBATCH_PARTITION` can override the helper defaults if your allocation differs

If you had to clean up home storage, deleting `.mamba` and generic cache directories is usually recoverable. The important thing is to keep future Hugging Face and torch caches on VAST scratch so MusicGen downloads do not repopulate home.

After extraction, train one SAE per selected layer directory. The CHPC training script defaults to a smaller profile and reads from scratch-based feature directories.

Train `layer_final` on CHPC:

```bash
TRAIN_LAYER_NAME=layer_final \
sbatch Scripts/TrainingScripts/chpc_submit.slurm
```

Train a selected intermediate layer with a denser sample schedule:

```bash
TRAIN_LAYER_NAME=layer_08 \
TRAIN_FRAME_STRIDE=2 \
TRAIN_BATCH_SIZE=64 \
sbatch Scripts/TrainingScripts/chpc_submit.slurm
```

Useful training overrides:

- `TRAIN_DATA_DIR` to point at a non-default feature directory
- `TRAIN_OUTPUT_DIR` to override the per-layer scratch output path
- `TRAIN_EPOCHS` defaults to `50`
- `TRAIN_BATCH_SIZE` defaults to `64`
- `TRAIN_FRAME_STRIDE` defaults to `4`
- `TRAIN_MAX_FRAMES` or `TRAIN_MAX_FILES` for pilot runs
- `TEMPORAL_MUSIC_ACTIVATIONS_PYTHON`, `TEMPORAL_MUSIC_ACTIVATIONS_CONDA_SH`, and `TEMPORAL_MUSIC_ACTIVATIONS_ENV_NAME` can override the Python bootstrap when Conda is installed in a non-standard location

**Analysis (Core Contribution):**

the goal is to characterize the shape of these time series and cluster features based on their temporal behavior.

to do this we need to create four types of profiles:

- **Static:** High, sustained activations throughout the track.
    - detectable through low variance over time, and a high mean.
    - should math original paper
- **Structural/Periodic:** activations that oscillate predictably.
    - detectable via auto correlation or a Fourier analysis of the time series.
- **Narrative Arc:** Monotonically increasing, single spike, or decaying profiles
    - Detectable by fitting simple trend models or using change-point detection algorithms.
- **Local Event:** brief, isolated activation pulses
    - Detectable by high kurtosis in the time series (the activation is near zero most of the time but spikes sharply at specific moments.)

References:

[1] https://www.google.com/url?q=https://arxiv.org/abs/2505.18186&sa=D&source=docs&ust=1772811583918737&usg=AOvVaw2Ec7i8IAEYYMlgYCigYgGC