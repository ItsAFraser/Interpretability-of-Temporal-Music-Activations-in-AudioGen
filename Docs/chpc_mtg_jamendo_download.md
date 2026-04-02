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

- Input: `/scratch/general/vast/$USER/mtg-jamendo/raw_30s/audio`
- Output: `/scratch/general/vast/$USER/sae_output/features`

Override explicitly if needed:

```bash
RAW_AUDIO_DIR=/scratch/general/vast/u1406806/mtg-jamendo/raw_30s/audio \
SCRATCH_FEATURES_DIR=/scratch/general/vast/u1406806/sae_output/features \
sbatch Scripts/TrainingScripts/chpc_extract.slurm
```
