#!/usr/bin/env python3
"""Build HTML report from an existing JSON file."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from report.generate_html import save_report_html  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description="Convert tracking_report.json to HTML")
    p.add_argument("json", type=Path, nargs="?", default=ROOT / "tracking_report.json")
    p.add_argument("-o", "--output", type=Path, default=None)
    args = p.parse_args()

    if not args.json.is_file():
        raise SystemExit(f"JSON not found: {args.json}")

    out = args.output or args.json.with_suffix(".html")
    payload = json.loads(args.json.read_text(encoding="utf-8"))
    save_report_html(payload, out)
    print(f"Saved {out}")
    print(f"Open: file://{out.resolve()}")


if __name__ == "__main__":
    main()
