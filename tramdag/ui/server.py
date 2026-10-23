"""Standalone browser editor (no Jupyter needed)::

    python -m tramdag.ui runs/exp1/configuration.json --data train.csv

or from Python ``tramdag.ui.serve("runs/exp1/configuration.json", data=train_df)``.
The page talks to a tiny local HTTP server (standard library only, bound to 127.0.0.1).
"""

import json
import os
import pathlib
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .backend import EditorBackend

_STATIC = pathlib.Path(__file__).parent / "static"

INDEX = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>TRAM-DAG editor</title>
<link rel="stylesheet" href="/static/editor.css">
<style>body{margin:0;padding:16px;background:#f6f8fa;font-family:system-ui,sans-serif}
@media (prefers-color-scheme: dark){body{background:#0d1117;color:#e6edf3}
.tde{--tde-bg:#0d1117;--tde-panel:#161b22;--tde-fg:#e6edf3;--tde-muted:#8b949e;--tde-border:#30363d;--tde-node:#161b22;--tde-accent:#2f81f7}}
</style></head>
<body><div id="app"></div>
<script type="module">
import editor from "/static/editor.js";
const state = await (await fetch("/api/state")).json();
const listeners = {};
const fire = (k) => (listeners["change:" + k] || []).forEach((f) => f());
const model = {
  get: (k) => state[k],
  set: (k, v) => { state[k] = v; fire(k); },
  on: (ev, cb) => { (listeners[ev] ||= []).push(cb); },
  off: () => {},
  save_changes: async () => {
    const r = await fetch("/api/spec", { method: "POST", body: JSON.stringify(state.spec) });
    model.set("report", await r.json());
  },
  send: async (msg) => {
    const r = await fetch("/api/msg", { method: "POST", body: JSON.stringify(msg) });
    const upd = await r.json();
    for (const [k, v] of Object.entries(upd)) model.set(k, v);
  },
};
state.height = Math.max(420, window.innerHeight - 300);
editor.render({ model, el: document.getElementById("app") });
</script></body></html>"""


def _handler(backend):
    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def _send(self, code, body, ctype="application/json"):
            data = body if isinstance(body, bytes) else body.encode()
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                return self._send(200, INDEX, "text/html; charset=utf-8")
            if self.path == "/api/state":
                with lock:
                    return self._send(200, json.dumps(backend.state()))
            if self.path.startswith("/static/"):
                name = self.path[len("/static/"):]
                f = _STATIC / name
                if f.parent == _STATIC and f.exists():
                    ctype = "text/javascript" if name.endswith(".js") else "text/css"
                    return self._send(200, f.read_bytes(), ctype)
            self._send(404, "{}")

        def do_POST(self):
            n = int(self.headers.get("Content-Length") or 0)
            payload = json.loads(self.rfile.read(n) or b"{}")
            with lock:
                if self.path == "/api/spec":
                    return self._send(200, json.dumps(backend.update_spec(payload)))
                if self.path == "/api/msg":
                    return self._send(200, json.dumps(backend.message(payload)))
            self._send(404, "{}")

    return Handler


def make_server(config_path=None, data=None, port=8765, cfg=None):
    from ..TramDagConfig import TramDagConfig

    if cfg is None:
        if config_path and os.path.exists(config_path):
            cfg = TramDagConfig.load_json(config_path)
        else:
            path = os.path.abspath(config_path or "configuration.json")
            exp_dir = os.path.dirname(path)
            name = os.path.basename(exp_dir)
            cfg = TramDagConfig({"experiment_name": name, "PATHS": {"EXPERIMENT_DIR": exp_dir, "DATA_PATH": exp_dir, "LOG_DIR": None}}, CONF_DICT_PATH=path)
    if isinstance(data, (str, pathlib.Path)):
        import pandas as pd

        data = pd.read_csv(data)
    backend = EditorBackend(cfg, df=data)
    return ThreadingHTTPServer(("127.0.0.1", port), _handler(backend)), backend


def serve(config_path=None, data=None, port=8765, open_browser=True):
    httpd, _ = make_server(config_path, data=data, port=port)
    url = f"http://127.0.0.1:{httpd.server_address[1]}/"
    print(f"TRAM-DAG editor running at {url}  (Ctrl+C to stop)")
    if open_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
