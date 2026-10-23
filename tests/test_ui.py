"""Editor backend, notebook widget and standalone server (front-end is tested with a browser in
``benchmarks/ui_e2e.py``)."""

import json
import threading
import urllib.request

import pytest

from tramdag import TramDagConfig
from tramdag.ui.backend import EditorBackend


@pytest.fixture
def cfg(tmp_path, mixed_df):
    return TramDagConfig.from_dataframe(mixed_df, "x1 -> x2", experiment_dir=str(tmp_path / "ui"))


def test_backend_validate_and_save(cfg, mixed_df):
    be = EditorBackend(cfg, df=mixed_df)
    st = be.state()
    assert {n["name"] for n in st["spec"]["nodes"]} == set(mixed_df.columns)
    assert st["spec"]["training"]["epochs"] == 100
    spec = st["spec"]
    spec["edges"].append({"source": "x2", "target": "x1", "term": "ls", "group": None, "model": None})
    rep = be.update_spec(spec)
    assert any("cycle" in e for e in rep["errors"]) and rep["dirty"]
    out = be.message({"type": "save"})
    assert out["report"]["errors"][0].startswith("not saved")
    spec["edges"].pop()
    spec["edges"].append({"source": "x1", "target": "x3", "term": "cs", "group": None, "model": None})
    rep = be.update_spec(spec)
    assert rep["errors"] == [] and rep["formulas"]["x3"] == "h(x3 | x1) = SI + cs(x1)"
    out = be.message({"type": "save"})
    assert out["report"]["saved_at"]
    on_disk = json.load(open(cfg.CONF_DICT_PATH))
    assert on_disk["nodes"]["x3"]["transformation_terms_in_h()"] == {"x1": "cs"}


def test_backend_text_and_layout(cfg):
    be = EditorBackend(cfg)
    out = be.message({"type": "edges_text", "text": "(x1, x2) -> x4 [cs]"})
    assert {e["source"] for e in out["spec"]["edges"] if e["target"] == "x4"} == {"x1", "x2"}
    out = be.message({"type": "auto_layout", "width": 800, "height": 500})
    xs = {n["name"]: n["x"] for n in out["spec"]["nodes"]}
    assert xs["x1"] < xs["x2"] < xs["x4"]


def test_widget_traits(cfg):
    pytest.importorskip("anywidget")
    from tramdag.ui.widget import DagEditor

    w = DagEditor(cfg)
    spec = dict(w.spec)
    spec["edges"] = spec["edges"] + [{"source": "x2", "target": "x5", "term": "ls", "group": None, "model": None}]
    w.spec = spec
    assert w.report["errors"] == [] and w.report["dirty"]
    w._on_msg(w, {"type": "save"}, [])
    assert w.report["saved_at"]
    assert "x2" in cfg.conf_dict["nodes"]["x5"]["parents"]


def test_server_endpoints(cfg):
    from tramdag.ui.server import make_server

    httpd, _ = make_server(cfg=cfg, port=0)
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    base = f"http://127.0.0.1:{httpd.server_address[1]}"
    try:
        html = urllib.request.urlopen(base + "/").read().decode()
        assert "/static/editor.js" in html
        js = urllib.request.urlopen(base + "/static/editor.js").read().decode()
        assert "export default" in js
        state = json.loads(urllib.request.urlopen(base + "/api/state").read())
        spec = state["spec"]
        spec["edges"].append({"source": "x2", "target": "x3", "term": "ci", "group": None, "model": None})
        req = urllib.request.Request(base + "/api/spec", data=json.dumps(spec).encode(), method="POST")
        rep = json.loads(urllib.request.urlopen(req).read())
        assert rep["formulas"]["x3"] == "h(x3 | x2) = ci(x2)"
        req = urllib.request.Request(base + "/api/msg", data=json.dumps({"type": "save"}).encode(), method="POST")
        out = json.loads(urllib.request.urlopen(req).read())
        assert out["report"]["saved_at"]
        with pytest.raises(urllib.error.HTTPError):
            urllib.request.urlopen(base + "/static/../backend.py")
    finally:
        httpd.shutdown()
