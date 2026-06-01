#!/usr/bin/env python3
"""Build a tiny JSONL corpus from local text/markdown files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_dir", type=Path, help="Directory containing .txt or .md files.")
    parser.add_argument("output_jsonl", type=Path, help="Destination JSONL path.")
    args = parser.parse_args()

    files = sorted(
        path
        for path in args.source_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in {".txt", ".md"}
    )
    if not files:
        raise SystemExit(f"No .txt or .md files found under {args.source_dir}")

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.output_jsonl.open("w", encoding="utf-8") as handle:
        for index, path in enumerate(files, start=1):
            text = path.read_text(encoding="utf-8", errors="replace").strip()
            if not text:
                continue
            handle.write(
                json.dumps(
                    {
                        "id": f"doc-{index:05d}",
                        "text": text,
                        "source_path": str(path),
                    },
                    sort_keys=True,
                )
                + "\n"
            )
    print(f"wrote {args.output_jsonl}")


if __name__ == "__main__":
    main()
