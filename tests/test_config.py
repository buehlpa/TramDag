"""DagSpec / edge language / TramDagConfig construction and compatibility with existing configs."""

import glob
import json
import os

import numpy as np
import pandas as pd
import pytest

from tramdag import DagSpec, TramDagConfig
from tramdag.dag_spec import infer_node, normalize_data_type

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_edge_language_terms_and_groups():
    spec = DagSpec.from_edges(
        """
        x1 -> x2                # linear shift
        x1 -> x3 [cs]
        x2 -> x3 -> x4
        (x1, x2) -> x4 [cs]
        x4 : ordinal(3)
        """
    )
    assert spec.names == ["x1", "x2", "x3", "x4"]
    assert spec.term_codes("x3") == {"x1": "cs", "x2": "ls"}
    assert spec.term_codes("x4") == {"x1": "cs11", "x2": "cs12", "x3": "ls"}
    assert spec.node("x4").data_type == "ordinal_Xn_Yo" and spec.node("x4").levels == 3
    assert spec.formula("x4") == "h(x4 | x1, x2, x3) = SI + cs(x1, x2) + ls(x3)"


def test_edge_language_models_and_ci():
    spec = DagSpec.from_edges("a -> c [cs, model=MLP(16,16,tanh)]; (a, b) -> d [ci]; b -> c [ComplexShiftCustomTabular]")
    assert spec.find_edge("a", "c").model == "MLP(16,16,tanh)"
    assert spec.find_edge("b", "c").model == "ComplexShiftCustomTabular"
    assert spec.term_codes("d") == {"a": "ci11", "b": "ci12"}
    assert spec.validate() == ([], [])


@pytest.mark.parametrize(
    "text, msg",
    [
        ("a -> b; b -> a", "cycle"),
        ("a -> c [ci]; b -> c [ci]", "only one plain complex intercept"),
        ("(a, b) -> c [ls]", "interactions need a complex term"),
        ("a -> b [xx]", "unknown option"),
    ],
)
def test_invalid_graphs(text, msg):
    try:
        spec = DagSpec.from_edges(text)
    except ValueError as e:
        assert msg in str(e)
        return
    errors, _ = spec.validate()
    assert any(msg in e for e in errors), errors


def test_single_member_group_is_an_error():
    spec = DagSpec.from_edges("a -> c; b -> c")
    spec.add_edge("a", "c", term="cs", group=1)
    errors, _ = spec.validate()
    assert any("only one input" in e for e in errors)


def test_ordinal_needs_levels_and_reserved_names():
    spec = DagSpec.from_edges("a -> b", data_types={"b": "ordinal"})
    assert any("levels" in e for e in spec.validate()[0])
    spec2 = DagSpec.from_edges("a_U -> b")
    assert any("reserved" in e for e in spec2.validate()[0])


def test_unknown_model_is_reported():
    spec = DagSpec.from_edges("a -> b [cs, model=NoSuchNet]")
    assert any("Unknown model" in e for e in spec.validate()[0])


def test_topological_reordering_in_config():
    spec = DagSpec.from_edges("c -> a; b -> a; c -> b")
    conf = spec.to_conf_dict("t", "/tmp/t")
    assert list(conf["data_type"]) == ["c", "b", "a"]
    adj = np.array(conf["adj_matrix"])
    assert np.all(np.tril(adj != "0") == False)  # noqa: E712  upper triangular


@pytest.mark.parametrize("path", sorted(glob.glob(os.path.join(ROOT, "example_notebooks", "*", "configuration.json"))))
def test_existing_configs_roundtrip(path):
    conf = json.load(open(path))
    spec = DagSpec.from_conf_dict(conf)
    new = spec.to_conf_dict(conf["experiment_name"], conf["PATHS"]["EXPERIMENT_DIR"])
    assert new["adj_matrix"] == conf["adj_matrix"]
    assert new["model_names"] == conf["model_names"]
    assert new["data_type"] == conf["data_type"]
    for n, nc in conf["nodes"].items():
        for k in ("parents", "parents_datatype", "transformation_terms_in_h()", "transformation_term_nn_models_in_h()", "node_type"):
            assert new["nodes"][n][k] == nc[k], (n, k)


def test_dict_roundtrip_for_ui():
    spec = DagSpec.from_edges("(a, b) -> c [cs]; a -> b; c : ordinal(4)")
    spec.auto_layout()
    again = DagSpec.from_dict(json.loads(spec.to_json()))
    assert again.to_dict() == spec.to_dict()


def test_infer_node():
    assert infer_node(pd.Series([0.1, 2.3, 4.0], name="a")) == ("continous", None)
    assert infer_node(pd.Series([0, 1, 2, 1], name="b")) == ("ordinal_Xn_Yo", 3)
    assert infer_node(pd.Series([True, False], name="c")) == ("ordinal_Xn_Yo", 2)
    assert infer_node(pd.Series(range(50), name="d")) == ("continous", None)
    assert normalize_data_type("continuous") == "continous"
    assert normalize_data_type("ordinal") == "ordinal_Xn_Yo"


def test_from_dataframe_writes_complete_config(tmp_path, mixed_df):
    cfg = TramDagConfig.from_dataframe(
        mixed_df, "x1 -> x2; x1, x2 -> x3; (x1, x2) -> x4 [cs]; x3 -> x4; x4 -> x5",
        experiment_dir=str(tmp_path / "e"), training={"epochs": 7},
    )
    on_disk = json.load(open(tmp_path / "e" / "configuration.json"))
    assert on_disk["nodes"]["x3"]["levels"] == 4 and on_disk["nodes"]["x5"]["levels"] == 2
    assert on_disk["training"]["epochs"] == 7
    assert on_disk["experiment_name"] == "e"
    loaded = TramDagConfig.load_json(str(tmp_path / "e" / "configuration.json"))
    assert loaded.conf_dict["adj_matrix"] == cfg.conf_dict["adj_matrix"]


def test_config_editing_methods(tmp_path, mixed_df):
    cfg = TramDagConfig.from_dataframe(mixed_df, "x1 -> x2", experiment_dir=str(tmp_path / "e"))
    cfg.add_edges("x2 -> x4 [cs]")
    cfg.add_edge("x1", "x4")
    assert cfg.conf_dict["nodes"]["x4"]["transformation_terms_in_h()"] == {"x1": "ls", "x2": "cs"}
    cfg.remove_edge("x1", "x4")
    assert cfg.conf_dict["nodes"]["x4"]["parents"] == ["x2"]
    cfg.set_node("x3", data_type="ordinal_Xc_Yo")
    assert cfg.conf_dict["data_type"]["x3"] == "ordinal_Xc_Yo"
    cfg.set_training(epochs=3, learning_rate=0.05)
    assert json.load(open(cfg.CONF_DICT_PATH))["training"] == {"epochs": 3, "learning_rate": 0.05}
    with pytest.raises(ValueError):
        cfg.add_edges("x4 -> x1")  # creates a cycle


def test_training_defaults_from_config_are_used(tmp_path, mixed_df):
    from tramdag import TramDagModel

    cfg = TramDagConfig.from_dataframe(mixed_df[["x1", "x2"]], "x1 -> x2", experiment_dir=str(tmp_path / "e"),
                                       training={"epochs": 4, "batch_size": 256})
    m = TramDagModel.from_config(cfg)
    m.fit(mixed_df[["x1", "x2"]], verbose=False)
    assert len(m.loss_history()["x2"]["train"]) == 4
