"""Serve the built-in tabletop tier over the environment protocol (no simulator needed).

    python -m robolivemem.serve --port 8800

The semi-realistic and controlled tiers have their own servers (tiers/semi/server.py, tiers/controlled/server.py);
the realistic tier runs one BEHAVIOR-1K server per scene (tiers/realistic).
"""
import argparse

from .protocol import Bridge
from .tabletop import INFO, serve_runner


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8800)
    a = ap.parse_args(argv)
    Bridge(serve_runner, INFO).run(a.host, a.port)


if __name__ == "__main__":
    main()
