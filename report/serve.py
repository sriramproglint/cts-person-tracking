#!/usr/bin/env python3
"""Serve tracking report HTML/JSON on localhost."""

from __future__ import annotations

import argparse
import http.server
import os
import socketserver
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    p = argparse.ArgumentParser(description="Serve tracking reports on localhost")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--dir", type=Path, default=ROOT, help="Folder with .html / .json reports")
    args = p.parse_args()

    os.chdir(args.dir)
    with socketserver.TCPServer(("", args.port), http.server.SimpleHTTPRequestHandler) as httpd:
        print(f"Serving {args.dir.resolve()}")
        print(f"  http://localhost:{args.port}/my_report.html")
        print(f"  http://localhost:{args.port}/tracking_report.html")
        print("Ctrl+C to stop.")
        httpd.serve_forever()


if __name__ == "__main__":
    main()
