"""Deploy the serving system as a Hugging Face Docker Space.

One-time setup: `hf auth login` with a WRITE token (or set HF_TOKEN), then:

    python deploy/deploy_hf.py [--space realtime-recsys]

Stages the Space files (Space README + Dockerfile + src/ + artifacts/ +
requirements.txt) into a temp dir and uploads in one commit; large binaries go
through LFS automatically. Re-running updates the Space in place.
"""

from __future__ import annotations

import argparse
import shutil
import tempfile
from pathlib import Path

from huggingface_hub import HfApi

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def stage(tmp: Path) -> None:
    shutil.copy(PROJECT_ROOT / "deploy/hf_space/README.md", tmp / "README.md")
    shutil.copy(PROJECT_ROOT / "deploy/hf_space/Dockerfile", tmp / "Dockerfile")
    shutil.copy(PROJECT_ROOT / "requirements.txt", tmp / "requirements.txt")
    shutil.copytree(
        PROJECT_ROOT / "src", tmp / "src", ignore=shutil.ignore_patterns("__pycache__")
    )
    shutil.copytree(PROJECT_ROOT / "artifacts", tmp / "artifacts")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--space", default="realtime-recsys")
    parser.add_argument("--private", action="store_true")
    args = parser.parse_args()

    api = HfApi()
    user = api.whoami()["name"]
    repo_id = f"{user}/{args.space}"
    api.create_repo(
        repo_id, repo_type="space", space_sdk="docker", private=args.private, exist_ok=True
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        stage(tmp)
        api.upload_folder(
            repo_id=repo_id,
            repo_type="space",
            folder_path=tmp,
            commit_message="Deploy serving system",
        )
    print(f"deployed -> https://huggingface.co/spaces/{repo_id}")
    print("(first build takes a few minutes; watch the Space's Logs tab)")


if __name__ == "__main__":
    main()
