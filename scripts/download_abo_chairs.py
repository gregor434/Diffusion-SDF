#!/usr/bin/env python3
from __future__ import annotations

"""Download every filtered ABO chair GLB and write preprocessing metadata."""

import argparse
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import shutil
import time
import urllib.parse
import urllib.request


DEFAULT_CHAIRS_CSV = Path("datasets/ABO/abo_chair_models.csv")
DEFAULT_TARGET_ROOT = Path("datasets/ABO/models_chair_full")
DEFAULT_METADATA_OUT = Path("datasets/ABO/abo_chairs_full.json")
DEFAULT_BUCKET_BASE_URL = "https://amazon-berkeley-objects.s3.amazonaws.com"
DEFAULT_PREFIX = "3dmodels/original"


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be >= 1")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chairs-csv", type=Path, default=DEFAULT_CHAIRS_CSV)
    parser.add_argument("--target-root", type=Path, default=DEFAULT_TARGET_ROOT)
    parser.add_argument("--metadata-out", type=Path, default=DEFAULT_METADATA_OUT)
    parser.add_argument("--bucket-base-url", default=DEFAULT_BUCKET_BASE_URL)
    parser.add_argument("--prefix", default=DEFAULT_PREFIX)
    parser.add_argument("--max-workers", type=positive_int, default=16)
    parser.add_argument("--timeout", type=positive_int, default=300)
    parser.add_argument("--download-retries", type=positive_int, default=3)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--metadata-only", action="store_true")
    return parser.parse_args()


def load_chairs(
    chairs_csv: Path,
    target_root: Path,
    bucket_base_url: str,
    prefix: str,
) -> dict[str, dict[str, object]]:
    with chairs_csv.open(newline="", encoding="utf-8") as source:
        rows = list(csv.DictReader(source))
    if not rows:
        raise RuntimeError(f"chair CSV is empty: {chairs_csv}")

    products = {}
    for row in rows:
        model_id = str(row["3dmodel_id"])
        relative_path = str(row["path"]).lstrip("/")
        products[model_id] = {
            **row,
            "instance_id": model_id,
            "3dmodel_id": model_id,
            "product_type_key": "CHAIR",
            "abo_path": relative_path,
            "local_glb_path": str(target_root / f"{model_id}.glb"),
            "download_url": (
                f"{bucket_base_url.rstrip('/')}/{prefix.strip('/')}/"
                f"{urllib.parse.quote(relative_path, safe='/')}"
            ),
        }
    return products


def write_metadata(
    path: Path,
    chairs_csv: Path,
    products: dict[str, dict[str, object]],
) -> None:
    payload = {
        "dataset_key": "abo",
        "source": {"chairs_csv": str(chairs_csv)},
        "selection": {"product_types": ["CHAIR"]},
        "summary": {
            "num_products": len(products),
            "product_type_counts": {"CHAIR": len(products)},
        },
        "products": products,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def download_one(entry: dict[str, object], timeout: int, retries: int) -> str:
    target = Path(str(entry["local_glb_path"]))
    temporary = target.with_suffix(target.suffix + f".tmp.{os.getpid()}")
    last_error = None
    for attempt in range(1, retries + 1):
        try:
            request = urllib.request.Request(
                str(entry["download_url"]), headers={"User-Agent": "Mozilla/5.0"}
            )
            with urllib.request.urlopen(request, timeout=timeout) as response:
                with temporary.open("wb") as output:
                    shutil.copyfileobj(response, output)
            if temporary.stat().st_size <= 0:
                raise RuntimeError("downloaded an empty file")
            temporary.replace(target)
            return f"downloaded: {target}"
        except Exception as error:
            last_error = error
            temporary.unlink(missing_ok=True)
            if attempt < retries:
                time.sleep(attempt)
    raise RuntimeError(
        f"failed after {retries} attempts: {entry['download_url']}: {last_error}"
    )


def download_all(
    products: dict[str, dict[str, object]],
    skip_existing: bool,
    max_workers: int,
    timeout: int,
    retries: int,
) -> tuple[int, int]:
    pending = []
    skipped = 0
    for entry in products.values():
        target = Path(str(entry["local_glb_path"]))
        target.parent.mkdir(parents=True, exist_ok=True)
        if skip_existing and target.is_file() and target.stat().st_size > 0:
            skipped += 1
        else:
            pending.append(entry)

    failures = []
    downloaded = 0
    with ThreadPoolExecutor(max_workers=min(max_workers, len(pending) or 1)) as pool:
        futures = {
            pool.submit(download_one, entry, timeout, retries): entry
            for entry in pending
        }
        for future in as_completed(futures):
            entry = futures[future]
            try:
                print(future.result(), flush=True)
                downloaded += 1
            except Exception as error:
                failures.append(str(entry["3dmodel_id"]))
                print(f"failed: {entry['3dmodel_id']}: {error}", flush=True)
    if failures:
        raise RuntimeError(
            f"failed to download {len(failures)} chairs: {', '.join(failures)}"
        )
    return downloaded, skipped


def main() -> None:
    args = parse_args()
    products = load_chairs(
        args.chairs_csv, args.target_root, args.bucket_base_url, args.prefix
    )
    args.target_root.mkdir(parents=True, exist_ok=True)
    write_metadata(args.metadata_out, args.chairs_csv, products)
    print(f"selected chairs: {len(products)}", flush=True)
    if args.metadata_only:
        return
    downloaded, skipped = download_all(
        products,
        skip_existing=args.skip_existing,
        max_workers=args.max_workers,
        timeout=args.timeout,
        retries=args.download_retries,
    )
    print(f"summary: downloaded={downloaded} skipped={skipped}", flush=True)


if __name__ == "__main__":
    main()
