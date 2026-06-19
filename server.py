#!/usr/bin/env python3
"""Tiny static file server for the analyzer UI.

Run it from the project root:

    python3 server.py

Then open http://localhost:8000 in your browser.  The UI expects
``games.json`` to be present in the same directory (produced by
``download_games.py``).
"""
from __future__ import annotations

import argparse
from http.server import HTTPServer, SimpleHTTPRequestHandler


class NoCacheHandler(SimpleHTTPRequestHandler):
    """Disable caching so iterative dev edits show up on reload."""

    def end_headers(self) -> None:  # noqa: D401 - short override
        self.send_header("Cache-Control", "no-store")
        super().end_headers()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    server = HTTPServer((args.host, args.port), NoCacheHandler)
    print(f"Serving on http://{args.host}:{args.port}  (Ctrl+C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print()
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
