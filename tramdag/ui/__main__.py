import argparse

from .server import serve


def main():
    ap = argparse.ArgumentParser(prog="python -m tramdag.ui", description="Visual TRAM-DAG configuration editor")
    ap.add_argument("config", nargs="?", default="configuration.json", help="configuration.json to edit or create")
    ap.add_argument("--data", help="CSV file; its columns become nodes (types and levels are inferred)")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--no-browser", action="store_true")
    a = ap.parse_args()
    serve(a.config, data=a.data, port=a.port, open_browser=not a.no_browser)


if __name__ == "__main__":
    main()
