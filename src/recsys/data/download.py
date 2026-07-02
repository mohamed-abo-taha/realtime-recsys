"""Download MovieLens and convert to the project's raw schema.

NOTE: this hits the network. ml-latest-small is ~1 MB (quick iteration);
ml-25m is ~250 MB zipped (the real development dataset from the spec).
Nothing else in the project downloads anything.
"""

from __future__ import annotations

import argparse
import io
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

from recsys.config import data_dir

URLS = {
    "small": "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip",
    "25m": "https://files.grouplens.org/datasets/movielens/ml-25m.zip",
}


def download(variant: str, out_dir: Path) -> None:
    url = URLS[variant]
    print(f"downloading {url} ...")
    with urllib.request.urlopen(url) as resp:
        payload = resp.read()
    print(f"downloaded {len(payload) / 1e6:.1f} MB, extracting ...")

    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(payload)) as zf:
        names = zf.namelist()
        ratings_name = next(n for n in names if n.endswith("ratings.csv"))
        movies_name = next(n for n in names if n.endswith("movies.csv"))
        ratings = pd.read_csv(zf.open(ratings_name))
        movies = pd.read_csv(zf.open(movies_name))

    ratings = ratings.rename(columns={"userId": "user_id", "movieId": "item_id"})
    movies = movies.rename(columns={"movieId": "item_id"})
    ratings.to_parquet(out_dir / "ratings.parquet", index=False)
    movies.to_parquet(out_dir / "items.parquet", index=False)
    print(f"wrote {len(ratings)} ratings, {len(movies)} items -> {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("variant", choices=list(URLS), help="small (~1MB) or 25m (~250MB)")
    parser.add_argument("--out", type=Path, default=data_dir() / "raw")
    args = parser.parse_args()
    download(args.variant, args.out)


if __name__ == "__main__":
    main()
