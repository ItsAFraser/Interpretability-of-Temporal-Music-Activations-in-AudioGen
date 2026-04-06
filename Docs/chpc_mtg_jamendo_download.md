# CHPC MTG-Jamendo Download Guide (DTN + VAST)

This guide downloads MTG-Jamendo directly on CHPC transfer nodes to avoid login-node transfer load.

## Scope

- Full raw_30s audio (`--dataset raw_30s --type audio`)
- Destination: `/scratch/general/vast/u1406806/mtg-jamendo`
- Transfer host: CHPC external DTN (`dtn05` to `dtn08`)

## Policy and license notes

- CHPC recommends Data Transfer Nodes for large external transfers.
- `/scratch/general/vast` is not backed up and may be cleaned after inactivity.
- MTG-Jamendo is for non-commercial research and academic use.

## 1) Connect to a DTN

```bash
ssh u1406806@dtn05.chpc.utah.edu
```

Any `dtn05` to `dtn08` host is acceptable.

## 2) Go to your project repo and make script executable

```bash
cd /uufs/chpc.utah.edu/common/home/u1406806/CS6966_Interpretability_of_LLMs/Interpretability-of-Temporal-Music-Activations-in-AudioGen
chmod +x Scripts/SetupScripts/chpc_download_mtg_jamendo.sh
```

If your CHPC path differs, replace it with your actual repo location.

## 3) Run the full raw_30s audio download

```bash
Scripts/SetupScripts/chpc_download_mtg_jamendo.sh \
  --dataset raw_30s \
  --type audio \
  --source mtg-fast \
  --target-dir /scratch/general/vast/u1406806/mtg-jamendo \
  --yes
```

The script will:

- Check free space floor for your selection.
- Clone or update `MTG/mtg-jamendo-dataset`.
- Create a local virtual environment for the downloader.
- Run MTG's official `download.py` with `--unpack --remove`.
- Write a timestamped log under `/scratch/general/vast/u1406806/mtg-jamendo/logs`.

## 4) Resume behavior

If a transfer is interrupted, rerun the exact command from step 3. The MTG downloader supports reruns to fill missing pieces.

## 5) Basic validation

```bash
du -sh /scratch/general/vast/u1406806/mtg-jamendo
find /scratch/general/vast/u1406806/mtg-jamendo/raw_30s/audio -type f | wc -l
ls -1t /scratch/general/vast/u1406806/mtg-jamendo/logs | head
```

Optional quick decode check if `ffprobe` is available:

```bash
sample_file=$(find /scratch/general/vast/u1406806/mtg-jamendo/raw_30s/audio -type f -name '*.mp3' | head -n 1)
ffprobe "$sample_file"
```

## 6) Hook into feature extraction

The extraction Slurm script now defaults to:

- Account: `soc-gpu-np`
- Partition: `soc-gpu-np`
- Input: `/scratch/general/vast/$USER/mtg-jamendo/raw_30s/audio`
- Output: `/scratch/general/vast/$USER/sae_output/features`
- Layers: `0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,-1`
- Queue-friendly resources: `1 GPU`, `8 CPU`, `16 GB RAM`, `6 hours`

Before submitting extraction or training jobs, create and validate the CUDA environment once on the login node:

```bash
conda env create -f Temporal-Music-Activations-cuda.yaml
conda activate temporal-music-activations-cuda
python -c "import torch, transformers; print(torch.__version__)"
```

If your Conda install is not discoverable automatically, the batch scripts now support these explicit overrides:

- `TEMPORAL_MUSIC_ACTIVATIONS_PYTHON=/path/to/env/bin/python`
- `TEMPORAL_MUSIC_ACTIVATIONS_CONDA_SH=/path/to/etc/profile.d/conda.sh`
- `TEMPORAL_MUSIC_ACTIVATIONS_ENV_NAME=temporal-music-activations-cuda`

Run a pilot extraction chunk:

```bash
EXTRACT_CHUNK_SIZE=256 \
sbatch Scripts/TrainingScripts/chpc_extract.slurm
```

Run several chunks as a job array:

```bash
EXTRACT_CHUNK_SIZE=256 \
sbatch --array=0-7 Scripts/TrainingScripts/chpc_extract.slurm
```

For a more convenient first substantial run over all layers, use:

```bash
bash Scripts/TrainingScripts/submit_chunked_extract.sh
```

This helper defaults to 1,024 files split into 8 jobs of 128 files each. Override with `TOTAL_FILES` and `CHUNK_SIZE` if you want a larger or smaller first pass.

Example with an explicit Python override:

```bash
TEMPORAL_MUSIC_ACTIVATIONS_PYTHON=$HOME/miniforge3/envs/temporal-music-activations-cuda/bin/python \
bash Scripts/TrainingScripts/submit_chunked_extract.sh
```

If your allocation differs, override the helper submit target with:

```bash
SBATCH_ACCOUNT=<your-account> \
SBATCH_PARTITION=<your-partition> \
bash Scripts/TrainingScripts/submit_chunked_extract.sh
```

Each array task processes a deterministic slice of the sorted audio list using:

- `EXTRACT_FILE_START`, if set explicitly
- otherwise `SLURM_ARRAY_TASK_ID * EXTRACT_CHUNK_SIZE`

Override explicitly if needed:

```bash
RAW_AUDIO_DIR=/scratch/general/vast/u1406806/mtg-jamendo/raw_30s/audio \
SCRATCH_FEATURES_DIR=/scratch/general/vast/u1406806/sae_output/features \
sbatch Scripts/TrainingScripts/chpc_extract.slurm
```

## 7) Train SAE on selected layers with smaller SLURM jobs

The SAE training Slurm script is also sized down by default:

- Account: `soc-gpu-np`
- Partition: `soc-gpu-np`
- Input base: `/scratch/general/vast/$USER/sae_output/features`
- Output base: `/scratch/general/vast/$USER/sae_output/models`
- Queue-friendly resources: `1 GPU`, `8 CPU`, `16 GB RAM`, `6 hours`
- Training defaults: `epochs=50`, `batch_size=64`, `frame_stride=4`, `num_workers=8`

The training job uses the same Python bootstrap overrides as extraction, so you can pass the same `TEMPORAL_MUSIC_ACTIVATIONS_*` variables when submitting training jobs.

Train `layer_final`:

```bash
TRAIN_LAYER_NAME=layer_final \
sbatch Scripts/TrainingScripts/chpc_submit.slurm
```

Train a selected intermediate layer:

```bash
TRAIN_LAYER_NAME=layer_08 \
TRAIN_FRAME_STRIDE=2 \
TRAIN_BATCH_SIZE=64 \
sbatch Scripts/TrainingScripts/chpc_submit.slurm
```

Useful overrides:

- `TRAIN_DATA_DIR` to point at a custom feature directory
- `TRAIN_OUTPUT_DIR` to choose a custom model output directory
- `TRAIN_MAX_FRAMES` or `TRAIN_MAX_FILES` for pilot runs
- `TRAIN_EPOCHS` if a layer needs a longer run once the pilot is stable
