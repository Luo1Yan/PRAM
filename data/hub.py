"""Resolve benchmark-relative pack paths locally or on the Hugging Face Hub."""

from pathlib import Path


def resolve_pack_path(
    pack_file: str, data_dir: str | Path, hf_repo: str | None = None, revision: str = "main",
) -> Path:
    local = Path(data_dir) / pack_file
    if local.is_file():
        return local
    if hf_repo:
        from huggingface_hub import hf_hub_download

        return Path(hf_hub_download(
            repo_id=hf_repo, filename=pack_file, repo_type="dataset",
            revision=revision, local_dir=str(data_dir),
        ))
    raise FileNotFoundError(
        f"Missing feature pack {local}. Pass --data_dir with the pack directory "
        "or --hf_repo with its Hugging Face dataset ID."
    )
