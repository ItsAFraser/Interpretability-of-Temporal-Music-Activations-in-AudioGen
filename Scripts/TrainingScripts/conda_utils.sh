#!/usr/bin/env bash

temporal_die() {
  echo "Error: $*" >&2
  return 1
}

temporal_warn() {
  echo "Warning: $*" >&2
}

temporal_find_conda_sh() {
  if [[ -n "${TEMPORAL_MUSIC_ACTIVATIONS_CONDA_SH:-}" ]]; then
    if [[ -f "${TEMPORAL_MUSIC_ACTIVATIONS_CONDA_SH}" ]]; then
      printf '%s\n' "${TEMPORAL_MUSIC_ACTIVATIONS_CONDA_SH}"
      return 0
    fi
    temporal_die "TEMPORAL_MUSIC_ACTIVATIONS_CONDA_SH points to a missing file: ${TEMPORAL_MUSIC_ACTIVATIONS_CONDA_SH}"
    return 1
  fi

  if command -v conda >/dev/null 2>&1; then
    local conda_base
    conda_base="$(conda info --base 2>/dev/null || true)"
    if [[ -n "$conda_base" && -f "$conda_base/etc/profile.d/conda.sh" ]]; then
      printf '%s\n' "$conda_base/etc/profile.d/conda.sh"
      return 0
    fi
  fi

  local candidate
  for candidate in \
    "$HOME/miniconda3/etc/profile.d/conda.sh" \
    "$HOME/miniforge3/etc/profile.d/conda.sh" \
    "$HOME/anaconda3/etc/profile.d/conda.sh"
  do
    if [[ -f "$candidate" ]]; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done

  return 1
}

temporal_initialize_conda() {
  if command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)" >/dev/null 2>&1 || true
  fi

  if ! command -v conda >/dev/null 2>&1; then
    if command -v module >/dev/null 2>&1; then
      local module_name
      local module_candidates=()

      if [[ -n "${TEMPORAL_MUSIC_ACTIVATIONS_LOAD_CONDA_MODULE:-}" ]]; then
        module_candidates+=("${TEMPORAL_MUSIC_ACTIVATIONS_LOAD_CONDA_MODULE}")
      fi
      module_candidates+=(miniforge3/25.11.0 miniforge3 miniconda3/25.9.1 miniconda3)

      for module_name in "${module_candidates[@]}"; do
        module load "$module_name" >/dev/null 2>&1 || continue
        break
      done
    fi
  fi

  if ! command -v conda >/dev/null 2>&1; then
    local conda_sh
    conda_sh="$(temporal_find_conda_sh)" || {
      temporal_die "Unable to locate conda. Set TEMPORAL_MUSIC_ACTIVATIONS_PYTHON to an env python, or set TEMPORAL_MUSIC_ACTIVATIONS_CONDA_SH to your conda.sh path."
      return 1
    }
    # shellcheck disable=SC1090
    source "$conda_sh"
  fi

  if ! command -v conda >/dev/null 2>&1; then
    temporal_die "conda is still unavailable after initialization."
    return 1
  fi

  eval "$(conda shell.bash hook)" >/dev/null 2>&1 || {
    temporal_die "Failed to initialize the conda shell hook."
    return 1
  }
}

temporal_conda_env_exists() {
  local env_name="$1"
  conda env list | awk 'NR > 2 {print $1}' | grep -Fxq "$env_name"
}

temporal_select_env_name() {
  local requested_env="${TEMPORAL_MUSIC_ACTIVATIONS_ENV_NAME:-}"
  if [[ -n "$requested_env" ]]; then
    if temporal_conda_env_exists "$requested_env"; then
      printf '%s\n' "$requested_env"
      return 0
    fi
    temporal_die "Requested conda environment '$requested_env' was not found."
    return 1
  fi

  local env_name
  for env_name in "$@"; do
    [[ -z "$env_name" ]] && continue
    if temporal_conda_env_exists "$env_name"; then
      printf '%s\n' "$env_name"
      return 0
    fi
  done

  temporal_die "None of the expected conda environments were found: $*"
  return 1
}

temporal_setup_python_environment() {
  if [[ -n "${TEMPORAL_MUSIC_ACTIVATIONS_PYTHON:-}" ]]; then
    if ! command -v "${TEMPORAL_MUSIC_ACTIVATIONS_PYTHON}" >/dev/null 2>&1 && [[ ! -x "${TEMPORAL_MUSIC_ACTIVATIONS_PYTHON}" ]]; then
      temporal_die "TEMPORAL_MUSIC_ACTIVATIONS_PYTHON is not executable: ${TEMPORAL_MUSIC_ACTIVATIONS_PYTHON}"
      return 1
    fi
    export TEMPORAL_MUSIC_ACTIVATIONS_ACTIVE_ENV="external-python"
    export TEMPORAL_MUSIC_ACTIVATIONS_ACTIVE_PYTHON="${TEMPORAL_MUSIC_ACTIVATIONS_PYTHON}"
    return 0
  fi

  temporal_initialize_conda || return 1

  local env_prefix="${TEMPORAL_MUSIC_ACTIVATIONS_ENV_PREFIX:-}"
  if [[ -n "$env_prefix" ]]; then
    if [[ ! -x "$env_prefix/bin/python" ]]; then
      temporal_die "TEMPORAL_MUSIC_ACTIVATIONS_ENV_PREFIX does not contain an executable python at $env_prefix/bin/python"
      return 1
    fi

    conda activate "$env_prefix" >/dev/null 2>&1 || {
      temporal_die "Failed to activate conda environment prefix '$env_prefix'."
      return 1
    }

    export TEMPORAL_MUSIC_ACTIVATIONS_ACTIVE_ENV="$env_prefix"
    export TEMPORAL_MUSIC_ACTIVATIONS_ACTIVE_PYTHON="$env_prefix/bin/python"

    if [[ -n "${CONDA_PREFIX:-}" ]]; then
      export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
    fi
    return 0
  fi

  local env_name
  env_name="$(temporal_select_env_name "$@")" || return 1
  conda activate "$env_name" >/dev/null 2>&1 || {
    temporal_die "Failed to activate conda environment '$env_name'."
    return 1
  }

  export TEMPORAL_MUSIC_ACTIVATIONS_ACTIVE_ENV="$env_name"
  if [[ -z "${CONDA_PREFIX:-}" || ! -x "${CONDA_PREFIX}/bin/python" ]]; then
    temporal_die "Activated conda environment '$env_name' does not provide an executable python at \
${CONDA_PREFIX:-<unset>}/bin/python"
    return 1
  fi
  export TEMPORAL_MUSIC_ACTIVATIONS_ACTIVE_PYTHON="${CONDA_PREFIX}/bin/python"

  if [[ -n "${CONDA_PREFIX:-}" ]]; then
    export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
  fi
}

temporal_print_python_diagnostics() {
  local python_bin="${1:-${TEMPORAL_MUSIC_ACTIVATIONS_ACTIVE_PYTHON:-}}"
  if [[ -z "$python_bin" ]]; then
    temporal_die "No active Python executable is available for diagnostics."
    return 1
  fi

  echo "Resolved Python environment: ${TEMPORAL_MUSIC_ACTIVATIONS_ACTIVE_ENV:-unknown}"
  echo "Resolved Python executable: $python_bin"
  "$python_bin" --version
}

temporal_validate_python_imports() {
  local python_bin="$1"
  shift

  if [[ -z "$python_bin" ]]; then
    temporal_die "No Python executable provided for import validation."
    return 1
  fi

  "$python_bin" - "$@" <<'PY'
import importlib.util
import sys

missing = [name for name in sys.argv[1:] if importlib.util.find_spec(name) is None]
if missing:
    raise SystemExit("Missing Python modules: " + ", ".join(missing))

print("Validated Python imports: " + ", ".join(sys.argv[1:]))
PY
}

temporal_print_torch_device_diagnostics() {
  local python_bin="${1:-${TEMPORAL_MUSIC_ACTIVATIONS_ACTIVE_PYTHON:-}}"
  if [[ -z "$python_bin" ]]; then
    temporal_die "No active Python executable is available for torch diagnostics."
    return 1
  fi

  "$python_bin" - <<'PY'
import torch

print(f"Torch version: {torch.__version__}")
print(f"torch.version.cuda: {torch.version.cuda}")
print(f"torch.backends.cuda.is_built(): {torch.backends.cuda.is_built()}")
print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
print(f"CUDA device count: {torch.cuda.device_count()}")
if torch.cuda.is_available() and torch.cuda.device_count() > 0:
    print(f"CUDA device 0: {torch.cuda.get_device_name(0)}")
PY
}

temporal_load_chpc_cuda_modules() {
  if ! command -v module >/dev/null 2>&1; then
    temporal_die "Environment modules are unavailable in this shell."
    return 1
  fi

  local gcc_module="${TEMPORAL_MUSIC_ACTIVATIONS_GCC_MODULE:-gcc}"
  local cuda_module="${TEMPORAL_MUSIC_ACTIVATIONS_CUDA_MODULE:-cuda/12.6.3}"
  echo "Loading CHPC compiler/CUDA modules..."

  if ! module --force purge >/dev/null 2>&1; then
    temporal_warn "module --force purge reported a problem; continuing with explicit loads."
  fi

  echo "  loading GCC module: $gcc_module"
  module load "$gcc_module" >/dev/null 2>&1 || {
    temporal_die "Failed to load GCC module '$gcc_module'."
    return 1
  }

  echo "  loading CUDA module: $cuda_module"
  module load "$cuda_module" >/dev/null 2>&1 || {
    temporal_die "Failed to load CUDA module '$cuda_module'. Override with TEMPORAL_MUSIC_ACTIVATIONS_CUDA_MODULE if needed."
    return 1
  }

  echo "Loaded modules:"
  module list 2>&1
}