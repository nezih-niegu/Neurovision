"""
neurovision.cli
==============
One entry point, two subcommands:

    uv run neurovision precompute [options]   cache the per-recording MI matrices
    uv run neurovision app        [options]   open the explorer in a browser
    uv run neurovision mine       [options]   MINE + channel-reconstruction benchmark

Both find the dataset and the results file on their own when run from inside the
project, so the common case needs no paths at all.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

DEFAULT_RESULTS = "results/mi_results.npz"


def project_root(start: Path | None = None) -> Path:
    """Nearest ancestor holding a pyproject.toml, else the current directory."""
    here = (start or Path.cwd()).resolve()
    for d in [here, *here.parents]:
        if (d / "pyproject.toml").exists():
            return d
    return here


def resolve_results(path: str | None) -> Path:
    return Path(path) if path else project_root() / DEFAULT_RESULTS


def cmd_precompute(argv: list[str]) -> int:
    from neurovision import precompute

    if not any(a.startswith("--out") for a in argv):
        out = resolve_results(None)
        out.parent.mkdir(parents=True, exist_ok=True)
        argv = [*argv, "--out", str(out)]
    precompute.main(argv)
    return 0


def cmd_mine(argv: list[str]) -> int:
    try:
        import torch  # noqa: F401
    except ImportError:
        print("The mine subcommand needs PyTorch, which is an optional extra "
              "because it is a ~1 GB download.\n\n    uv sync --extra mine\n",
              file=sys.stderr)
        return 1
    from neurovision import benchmark

    if not any(a.startswith("--out") for a in argv):
        out = project_root() / "results" / "benchmark"
        out.parent.mkdir(parents=True, exist_ok=True)
        argv = [*argv, "--out", str(out)]
    benchmark.main(argv)
    return 0


def cmd_app(argv: list[str]) -> int:
    p = argparse.ArgumentParser(prog="neurovision app", add_help=True)
    p.add_argument("--results", default=None,
                   help=f"cached .npz (default: {DEFAULT_RESULTS})")
    p.add_argument("--port", type=int, default=8501)
    args, extra = p.parse_known_args(argv)

    results = resolve_results(args.results)
    if not results.exists():
        print(f"No results at {results}\n"
              f"Run  uv run neurovision precompute  first.", file=sys.stderr)
        return 1

    app_py = Path(__file__).with_name("app.py")
    cmd = [sys.executable, "-m", "streamlit", "run", str(app_py),
           "--server.port", str(args.port), *extra,
           "--", "--results", str(results)]
    return subprocess.call(cmd)


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(
        prog="neurovision",
        description="Channel-wise mutual information for resting-state EEG.")
    parser.add_argument("command", choices=["precompute", "app", "mine"],
                        help="precompute: cache MI matrices · app: open the "
                             "explorer · mine: neural MI + prediction benchmark")
    if not argv:
        parser.print_help()
        return 1
    args = parser.parse_args(argv[:1])
    rest = argv[1:]
    return {"precompute": cmd_precompute, "app": cmd_app,
            "mine": cmd_mine}[args.command](rest)


if __name__ == "__main__":
    raise SystemExit(main())
