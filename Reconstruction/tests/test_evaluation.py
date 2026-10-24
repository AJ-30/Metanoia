"""Unit tests for evaluation (composition, matching, structure checks, metrics, report).

Uses the same synthetic two-room plan as ``test_data_prep.py``:

    +-----------win----------+
    |  bedroom  |   living   |
    |          door          |
    |           |            |
    +------------------------+

Run from the repository root:
    python -m pytest Reconstruction/tests -q
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from shapely.geometry import box

from Reconstruction.common.config_loader import load_params
from Reconstruction.data_prep.geometry import PlanGeometry
from Reconstruction.data_prep.missing import EASY, HARD, CaseSpec, MissingComponentGenerator
from Reconstruction.data_prep.plan_store import PlanStore, write_split
from Reconstruction.data_prep.rasterize import rasterize_plan
from Reconstruction.data_prep.schema import Schema
from Reconstruction.evaluation.baselines import predict_empty, predict_oracle, predict_rules
from Reconstruction.evaluation.case_metrics import prepare_case, score_samples
from Reconstruction.evaluation.components import extract_components, iou_matrix, match
from Reconstruction.evaluation.labels import EMPTY, LabelMaps, added, compose, semantic_to_labels
from Reconstruction.evaluation.report import build_tables
from Reconstruction.evaluation.structure_checks import PlanStructure

ROOT: Path = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def data_params():
    """Data-preparation parameters."""
    return load_params(ROOT / "data_prep" / "params.toml")


@pytest.fixture(scope="module")
def eval_params():
    """Evaluation parameters."""
    return load_params(ROOT / "evaluation" / "params.toml")


@pytest.fixture(scope="module")
def schema(data_params) -> Schema:
    """Project taxonomy."""
    return Schema.from_params(data_params)


@pytest.fixture(scope="module")
def store(tmp_path_factory, schema, data_params) -> PlanStore:
    """One-plan store with the synthetic plan."""
    door = box(48, 25, 52, 35)
    window = box(20, 56, 35, 60)
    walls = box(0, 0, 100, 60).difference(box(4, 4, 96, 56)).union(box(48, 4, 52, 56)).difference(door).difference(window)
    instances = [("bedroom", box(4, 4, 48, 56)), ("living", box(52, 4, 96, 56)), ("door", door), ("window", window)]
    geometry = PlanGeometry(plan_id=7, metres_per_unit=0.1, wall=walls, instances=instances)
    split_dir: Path = tmp_path_factory.mktemp("processed") / "test"
    write_split(split_dir, [rasterize_plan(geometry, schema, data_params)])
    return PlanStore(split_dir)


def _case(store, schema, data_params, cls: str, mode: str):
    """Build the case removing the first instance of ``cls``."""
    record = store.record(0)
    inst = next(i for i in record["instances"] if i["cls"] == cls)
    ctype = {"bedroom": "room", "living": "room"}.get(cls, cls)
    spec = CaseSpec("test", 0, 7, ctype, inst["layer"], inst["id"], mode, 0)
    return MissingComponentGenerator(schema, data_params, None).build_case(store.layers(0), record, spec)


def _labels(structure: list[list[int]], room: list[list[int]]) -> LabelMaps:
    """Small LabelMaps from nested lists."""
    return LabelMaps(np.array(structure, np.int8), np.array(room, np.int8))


def test_compose_lets_openings_override_walls_but_never_erases():
    # Arrange: structure classes are [wall=0, railing, column, window=3, door=4].
    base = _labels([[0, 4]], [[EMPTY, 1]])
    prediction = _labels([[4, 0]], [[2, 2]])
    # Act
    composed = compose(base, prediction)
    # Assert: door replaces wall; wall cannot replace door; room only fills empty.
    assert composed.structure.tolist() == [[4, 4]]
    assert composed.room.tolist() == [[2, 1]]
    assert added(base, composed).structure.tolist() == [[4, EMPTY]]


def test_matching_is_one_to_one_and_same_class(schema):
    # Two door pieces predicted, one true door overlapping the first half-way.
    truth = extract_components(_labels([[4, 4, EMPTY, EMPTY]] * 2, [[EMPTY] * 4] * 2), schema, 1)
    pred = extract_components(_labels([[EMPTY, 4, EMPTY, 3]] * 2, [[EMPTY] * 4] * 2), schema, 1)
    ious = iou_matrix(pred, truth)
    pairs = match(ious)
    assert len(pairs) == 1 and pairs[0][2] == pytest.approx(0.5)
    window_row = next(i for i, c in enumerate(pred) if c.cls == "window")
    assert ious[window_row].max() == 0.0


@pytest.mark.parametrize("cls,mode", [("door", EASY), ("door", HARD), ("window", EASY), ("wall", HARD), ("bedroom", HARD)])
def test_oracle_scores_perfectly(store, schema, data_params, eval_params, cls, mode):
    case = _case(store, schema, data_params, cls, mode)
    scores = score_samples(prepare_case(case, schema, eval_params), predict_oracle(case, schema, eval_params), schema, eval_params)
    assert scores["primary_detected"] and scores["primary_iou"] == pytest.approx(1.0)
    assert scores["tp@0.5"] == scores["n_truth"] == scores["n_pred"]
    assert scores["type_correct"] and scores["shell_pieces_diff"] == 0
    if cls in ("door", "window"):
        assert scores["connection_correct"]


def test_empty_prediction_scores_zero(store, schema, data_params, eval_params):
    case = _case(store, schema, data_params, "door", EASY)
    scores = score_samples(prepare_case(case, schema, eval_params), predict_empty(case, schema, eval_params), schema, eval_params)
    assert scores["n_pred"] == 0 and scores["tp@0.5"] == 0 and not scores["primary_detected"]
    assert scores["predicted_type"] == "none" and not scores["connection_correct"]


def test_missing_door_breaks_reachability_and_is_a_clue(store, schema, data_params, eval_params):
    case = _case(store, schema, data_params, "door", HARD)
    full = PlanStructure(semantic_to_labels(case.full, schema, 0.5), schema, eval_params)
    incomplete = PlanStructure(semantic_to_labels(case.incomplete, schema, 0.5), schema, eval_params)
    assert full.rooms_reachable() == (2, 2)
    assert incomplete.rooms_reachable() == (1, 2)
    assert prepare_case(case, schema, eval_params).fields["clue"]


def test_rules_baseline_restores_easy_door_gap(store, schema, data_params, eval_params):
    case = _case(store, schema, data_params, "door", EASY)
    scores = score_samples(prepare_case(case, schema, eval_params), predict_rules(case, schema, eval_params), schema, eval_params)
    assert scores["primary_detected"] and scores["type_correct"]


def test_best_of_k_picks_the_right_sample(store, schema, data_params, eval_params):
    case = _case(store, schema, data_params, "door", EASY)
    samples = np.concatenate([predict_empty(case, schema, eval_params), predict_oracle(case, schema, eval_params)])
    scores = score_samples(prepare_case(case, schema, eval_params), samples, schema, eval_params)
    assert scores["k"] == 2 and not scores["primary_detected"] and scores["bestk_primary_detected"]


def test_report_uses_micro_averaged_detection():
    records = pd.DataFrame({
        "comp_type": ["door", "door"], "mode": ["easy", "easy"], "margin_px": [0, 0],
        "size_bin": ["small", "large"], "location": ["interior", "interior"],
        "n_pred": [1, 3], "n_truth": [1, 1], "tp@0.5": [1, 1], "shell_pieces_diff": [0, 1],
    })
    overall = build_tables(records, [0.5])["overall"].iloc[0]
    assert overall["precision@0.5"] == pytest.approx(2 / 4)
    assert overall["recall@0.5"] == pytest.approx(1.0)
    assert overall["shell_exact"] == pytest.approx(0.5)
