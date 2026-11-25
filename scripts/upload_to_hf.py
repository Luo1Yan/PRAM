"""Upload numeric feature packs while preserving their benchmark-relative paths.

Use --folder data_cache for a directory containing packs/main and packs/sensing.
Use --path_in_repo packs when the selected folder contains main and sensing.
Authentication uses the Hugging Face Hub's configured credentials.
"""

import argparse
from pathlib import Path

from huggingface_hub import HfApi


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folder", required=True, type=Path)
    parser.add_argument("--repo_id", required=True)
    parser.add_argument("--path_in_repo", default="")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--private", action="store_true")
    args = parser.parse_args(argv)
    if not args.folder.is_dir() or not any(
        p.is_file() and p.suffix in (".npz", ".pt") for p in args.folder.rglob("*")
    ):
        parser.error("--folder must contain feature packs")
    api = HfApi()
    api.create_repo(repo_id=args.repo_id, repo_type="dataset", private=args.private, exist_ok=True)
    if args.revision != "main":
        api.create_branch(repo_id=args.repo_id, branch=args.revision, repo_type="dataset", exist_ok=True)
    commit = api.upload_folder(
        repo_id=args.repo_id, repo_type="dataset", revision=args.revision,
        folder_path=str(args.folder), path_in_repo=args.path_in_repo,
        allow_patterns=["*.npz", "*.pt", "README.md"], commit_message="Publish frozen relation packs",
    )
    print(commit.commit_url)
    return commit


if __name__ == "__main__":
    main()
