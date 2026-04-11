import argparse
import json
import os

from FullLengthAudioDataset import FullLengthAudioDataset


def parse_args():
    parser = argparse.ArgumentParser(
        description="Resolve a deterministic feature-file subset and write a relative-path manifest."
    )
    parser.add_argument("--data_dir", required=True, help="Layer feature directory to scan.")
    parser.add_argument(
        "--manifest_path",
        required=True,
        help="Text file that will receive one selected relative file path per line.",
    )
    parser.add_argument(
        "--metadata_path",
        default=None,
        help="Optional JSON metadata path describing the resolved subset.",
    )
    parser.add_argument("--max_files", type=int, default=0, help="Ordered file cap before selection.")
    parser.add_argument(
        "--random_subset_files",
        type=int,
        default=0,
        help="Randomly choose this many files after sorting (0 = all files).",
    )
    parser.add_argument("--subset_seed", type=int, default=42, help="Deterministic subset seed.")
    return parser.parse_args()


def main():
    args = parse_args()

    dataset = FullLengthAudioDataset(
        args.data_dir,
        max_files=args.max_files,
        random_subset_files=args.random_subset_files,
        subset_seed=args.subset_seed,
        sample_mode="mean",
    )

    selected_files = dataset.get_selected_relative_files()
    os.makedirs(os.path.dirname(args.manifest_path), exist_ok=True)
    with open(args.manifest_path, "w", encoding="utf-8") as handle:
        for relative_path in selected_files:
            handle.write(f"{relative_path}\n")

    if args.metadata_path:
        os.makedirs(os.path.dirname(args.metadata_path), exist_ok=True)
        metadata = {
            "data_dir": os.path.abspath(args.data_dir),
            "manifest_path": os.path.abspath(args.manifest_path),
            "max_files": int(args.max_files),
            "random_subset_files": int(args.random_subset_files),
            "subset_seed": int(args.subset_seed),
            "total_files_discovered": int(dataset.total_files_discovered),
            "selected_files": int(dataset.selected_files),
        }
        with open(args.metadata_path, "w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2)

    print(
        f"Resolved {dataset.selected_files} feature files from {dataset.total_files_discovered} discovered under {args.data_dir}"
    )
    print(f"Wrote manifest: {args.manifest_path}")
    if args.metadata_path:
        print(f"Wrote metadata: {args.metadata_path}")


if __name__ == "__main__":
    main()