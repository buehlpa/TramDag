"""Browser end-to-end test of the visual editor inside JupyterLab (anywidget front-end).

Needs: jupyterlab, playwright (``python -m playwright install chromium``).
Starts a throw-away JupyterLab, opens a notebook that calls ``cfg.edit()``, draws an edge by
dragging, turns it into a complex shift, saves, and checks configuration.json. Screenshots are
written next to the notebook.

    python benchmarks/ui_e2e.py /path/to/workdir
"""
import json
import os
import socket
import subprocess
import sys
import time

import nbformat

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def main(work):
    os.makedirs(work, exist_ok=True)
    exp = os.path.join(work, "exp")
    nb = nbformat.v4.new_notebook()
    nb.cells = [
        nbformat.v4.new_code_cell(
            f"import sys; sys.path.insert(0, {ROOT!r})\n"
            "import numpy as np, pandas as pd\n"
            "from tramdag import TramDagConfig\n"
            "rng = np.random.default_rng(0)\n"
            "df = pd.DataFrame({'x1': rng.normal(size=500), 'x2': rng.normal(size=500), 'x3': rng.integers(0, 3, 500)})\n"
            f"cfg = TramDagConfig.from_dataframe(df, 'x1 -> x2', experiment_dir={exp!r})\n"
            "cfg.edit()"
        )
    ]
    nbformat.write(nb, os.path.join(work, "editor.ipynb"))
    port = free_port()
    env = dict(os.environ, JUPYTER_CONFIG_DIR=os.path.join(work, ".jcfg"))
    proc = subprocess.Popen(
        [sys.executable, "-m", "jupyter", "lab", "--no-browser", f"--port={port}", "--ServerApp.token=",
         "--ServerApp.password=", "--ServerApp.disable_check_xsrf=True", f"--ServerApp.root_dir={work}"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=env,
    )
    try:
        from playwright.sync_api import sync_playwright

        for _ in range(60):
            try:
                socket.create_connection(("127.0.0.1", port), timeout=1).close()
                break
            except OSError:
                time.sleep(0.5)
        with sync_playwright() as p:
            b = p.chromium.launch()
            page = b.new_page(viewport={"width": 1400, "height": 950})
            page.goto(f"http://127.0.0.1:{port}/lab/tree/editor.ipynb")
            page.wait_for_selector(".jp-Notebook", timeout=60000)
            page.wait_for_timeout(3000)
            for _ in range(5):  # kernel picker / news notifications
                dlg = page.query_selector(".jp-Dialog .jp-mod-accept")
                if not dlg:
                    break
                dlg.click()
                page.wait_for_timeout(1000)
            page.click(".jp-Cell .jp-InputArea-editor")
            page.keyboard.press("Shift+Enter")
            page.wait_for_selector(".tde-node", timeout=120000)
            page.wait_for_timeout(800)
            page.screenshot(path=os.path.join(work, "jupyter_editor_1.png"))

            def node(name):
                for g in page.query_selector_all(".tde-node"):
                    if g.query_selector(".tde-node-name").text_content() == name:
                        return g

            def mid(el):
                bb = el.bounding_box()
                return bb["x"] + bb["width"] / 2, bb["y"] + bb["height"] / 2

            x, y = mid(node("x2").query_selector(".tde-port"))
            tx, ty = mid(node("x3").query_selector(".tde-node-box"))
            page.mouse.move(x, y)
            page.mouse.down()
            page.mouse.move(tx, ty, steps=10)
            page.mouse.up()
            page.wait_for_timeout(500)
            page.click(".tde-inspector .tde-term-name:text-is(\"Complex shift\")")
            page.wait_for_timeout(500)
            page.click(".tde-header button:text-is(\"Save\")")
            page.wait_for_timeout(1500)
            page.screenshot(path=os.path.join(work, "jupyter_editor_2.png"))
            status = page.inner_text(".tde-status")
            saved = page.inner_text(".tde-saveinfo")
            b.close()
        conf = json.load(open(os.path.join(exp, "configuration.json")))
        terms = conf["nodes"]["x3"]["transformation_terms_in_h()"]
        print("status:", status, "|", saved, "| x3 terms:", terms)
        assert terms == {"x2": "cs"}, terms
        print("OK")
    finally:
        proc.terminate()
        proc.wait(10)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, ".ui_e2e"))
