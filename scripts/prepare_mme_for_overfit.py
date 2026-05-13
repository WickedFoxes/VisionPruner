#!/usr/bin/env python3
"""Download lmms-lab/MME and convert it to LLaVA training JSON."""

from __future__ import annotations

import argparse
import io
import json
import re
from pathlib import Path
from typing import Any


IMAGE_EXTENSIONS = {
    "JPEG": ".jpg",
    "JPG": ".jpg",
    "PNG": ".png",
    "BMP": ".bmp",
    "WEBP": ".webp",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare lmms-lab/MME under playground/data for test_overfit.sh."
    )
    parser.add_argument("--repo-id", default="lmms-lab/MME")
    parser.add_argument("--split", default="test")
    parser.add_argument("--data-root", default="playground/data")
    parser.add_argument("--image-dir", default="mme/images")
    parser.add_argument("--json-path", default="mme_test.json")
    parser.add_argument("--dummy-json-path", default="mme_dummy_test.json")
    parser.add_argument("--dummy-size", type=int, default=16)
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Only export the first N samples. By default, export the full split.",
    )
    parser.add_argument(
        "--overwrite-images",
        action="store_true",
        help="Rewrite image files even when they already exist.",
    )
    return parser.parse_args()


def safe_stem(value: Any, fallback: str) -> str:
    text = str(value or fallback)
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("._")
    return text or fallback


def load_pil_image(raw_image: Any) -> Any:
    from PIL import Image

    if isinstance(raw_image, dict):
        if raw_image.get("bytes") is not None:
            return Image.open(io.BytesIO(raw_image["bytes"]))
        if raw_image.get("path") is not None:
            return Image.open(raw_image["path"])
    return raw_image


def image_extension(image: Any, raw_image: Any) -> str:
    if isinstance(raw_image, dict) and raw_image.get("path"):
        suffix = Path(raw_image["path"]).suffix
        if suffix:
            return suffix
    fmt = getattr(image, "format", None)
    if fmt:
        return IMAGE_EXTENSIONS.get(str(fmt).upper(), f".{str(fmt).lower()}")
    return ".jpg"


def save_image(image: Any, path: Path, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    if getattr(image, "mode", None) not in (None, "RGB"):
        image = image.convert("RGB")
    image.save(path)


def to_llava_record(row: dict[str, Any], image_name: str, index: int) -> dict[str, Any]:
    question_id = row.get("question_id") or f"mme_{index:06d}"
    question = str(row["question"]).strip()
    answer = str(row["answer"]).strip()
    return {
        "id": str(question_id),
        "image": image_name,
        "conversations": [
            {"from": "human", "value": f"<image>\n{question}"},
            {"from": "gpt", "value": answer},
        ],
    }


def main() -> None:
    args = parse_args()

    try:
        from datasets import Image, load_dataset
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: datasets. Install it with "
            "`pip install datasets pillow` in the environment used for this repo."
        ) from exc

    data_root = Path(args.data_root)
    image_root = data_root / args.image_dir
    json_path = data_root / args.json_path
    dummy_json_path = data_root / args.dummy_json_path

    dataset = load_dataset(args.repo_id, split=args.split)
    dataset = dataset.cast_column("image", Image(decode=False))

    records: list[dict[str, Any]] = []
    total = len(dataset) if args.max_samples is None else min(args.max_samples, len(dataset))

    for index, row in enumerate(dataset):
        if args.max_samples is not None and index >= args.max_samples:
            break

        raw_image = row["image"]
        image = load_pil_image(raw_image)
        category = safe_stem(row.get("category"), "uncategorized")
        stem = safe_stem(row.get("question_id"), f"mme_{index:06d}")
        extension = image_extension(image, raw_image)
        if stem.lower().endswith(extension.lower()):
            stem = stem[: -len(extension)]
        image_name = f"{category}/{stem}{extension}"
        save_image(image, image_root / image_name, args.overwrite_images)
        records.append(to_llava_record(row, image_name, index))

        if (index + 1) % 100 == 0 or index + 1 == total:
            print(f"exported {index + 1}/{total}", flush=True)

    data_root.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n")

    dummy_records = records[: args.dummy_size]
    dummy_json_path.write_text(json.dumps(dummy_records, ensure_ascii=False, indent=2) + "\n")

    print(f"wrote {len(records)} records to {json_path}")
    print(f"wrote {len(dummy_records)} records to {dummy_json_path}")
    print(f"saved images under {image_root}")


if __name__ == "__main__":
    main()
