"""Этап 11: модель КРН участков газопроводов (Мирзоев, Иващенко, Маршаков, 2016) с учётом
разницы магистральных и промысловых газопроводов и синтетичности данных."""

import copy
import shutil
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from field_twin import config, scc, storage
from field_twin.cycle import run_cycle
from field_twin.gateway.tag_dictionary import TagDictionary

S = scc.settings()
TAGS = TagDictionary.load()
NOW = pd.Timestamp("2026-07-01T00:00:00Z")


@pytest.fixture(scope="module")
def model():
    return scc.train(scc.synthetic_dataset(368, seed=0), S, scc.SYNTHETIC)


def test_architecture_from_article(model):
    assert len(model.inputs) == 10                       # 10 входных нейронов
    assert tuple(model.net.hidden_layer_sizes) == (13,)  # 1 скрытый слой из 13 нейронов
    assert model.net.activation == "logistic"            # сигмоида в скрытом слое
    assert model.net.out_activation_ == "identity"       # линейный выходной слой
    assert model.net.n_outputs_ == 2                     # два выхода


def test_split_like_article(model):
    m, v = model.metrics, model.validation_metrics
    assert m.no_scc_total + m.scc_total == 55 and v.no_scc_total + v.scc_total == 55


def test_metrics_report_marks_synthetic(model, capsys, tmp_path):
    text = model.metrics.to_text(S["reference"])
    assert "80.43" in text and "81.52" in text and "80.98" in text     # ориентиры табл. 2
    assert model.trained_on == "synthetic" and model.metrics.total_pct > 60
    scc.main(["--synthetic", "--out", str(tmp_path / "scc.joblib")])
    assert "СИНТЕТИКЕ" in capsys.readouterr().out


@pytest.mark.parametrize("p,name", [(0.05, "низкий"), (0.1, "умеренный"), (0.229, "умеренный"), (0.23, "средний"),
                                    (0.3, "высокий"), (0.4, "критичный"), (0.9, "критичный")])
def test_risk_classes(p, name):
    assert scc.risk_class(p, S["risk_classes"]) == name


def test_domain_shift_trunk_to_field(model):
    """Почему для промыслового «перенос не проверен»: на синтетике с иной связью факторов и КРН
    модель, обученная на магистральном, заметно хуже."""
    trunk = scc.synthetic_dataset(400, seed=5, pipeline_class="trunk")
    field = scc.synthetic_dataset(400, seed=5, pipeline_class="field")
    acc = lambda d: np.mean((model.probability(d) >= 0.5) == d["scc"])
    assert acc(field) < acc(trunk) - 0.1


def test_training_requires_source_label():
    with pytest.raises(ValueError):
        scc.train(scc.synthetic_dataset(120, seed=1), S, "unknown")


# ---- шурфы и применимость ----

def pits(conn, pipeline_segment, n, matches, p=0.8):
    t = datetime(2026, 7, 1, tzinfo=timezone.utc)
    for i in range(n):
        scc.add_pit(conn, TAGS, pipeline_segment, scc_found=i < matches, user_name="Сидоров", dug_at=t,
                    predicted_probability=p)


@pytest.mark.parametrize("n,matches,status", [(9, 9, "unverified"), (10, 8, "verified"), (10, 7, "rejected")])
def test_applicability_by_pits(n, matches, status):
    conn = storage.connect(":memory:")
    pits(conn, "MG-1-034", n, matches)
    a = scc.applicability(conn, S, TAGS)
    assert a["MG-1"].status == status and a["PL-01"].status == "unverified"


def test_add_pit_validation():
    conn = storage.connect(":memory:")
    with pytest.raises(ValueError):
        scc.add_pit(conn, TAGS, "NO-SUCH", True, "Сидоров", datetime(2026, 7, 1, tzinfo=timezone.utc))
    with pytest.raises(ValueError):
        scc.add_pit(conn, TAGS, "MG-1-034", True, "Сидоров", datetime(2026, 7, 1))


# ---- прогноз по участкам и П17 ----

def risk(p, cls="trunk", appl="unverified", synthetic=True):
    return scc.SegmentRisk("MG-1-034" if cls == "trunk" else "PL-01-001", "MG-1" if cls == "trunk" else "PL-01",
                           cls, p, scc.risk_class(p, S["risk_classes"]), NOW, cls == "trunk", appl, synthetic)


def test_p17_below_above():
    th = config.thresholds()
    assert not scc.check_p17([risk(0.29)], th, S, NOW)
    ev = scc.check_p17([risk(0.31)], th, S, NOW)
    assert len(ev) == 1 and ev[0].recipient == "chief_engineer" and "шурфе" in ev[0].message


def test_p17_caveats_synthetic_and_field():
    ev = scc.check_p17([risk(0.5, "field")], config.thresholds(), S, NOW)[0]
    assert ev.message.startswith("ПРЕДВАРИТЕЛЬНО")
    cav = ev.payload["caveats"]
    assert any("синтетических" in c for c in cav)
    assert any("промысловый" in c and "магистральных" in c for c in cav)
    assert any("шурфов" in c for c in cav)
    assert ev.payload["in_domain"] is False and ev.payload["synthetic_model"] is True


def test_p17_without_caveats_when_real_verified_trunk():
    ev = scc.check_p17([risk(0.5, "trunk", "verified", synthetic=False)], config.thresholds(), S, NOW)[0]
    assert not ev.message.startswith("ПРЕДВАРИТЕЛЬНО") and ev.payload["caveats"] == []


def test_p17_silent_where_rejected():
    assert scc.check_p17([risk(0.9, appl="rejected")], config.thresholds(), S, NOW) == []


def test_generator_writes_survey_per_segment(samples):
    df = pd.read_csv(samples.files["pipeline_segments"])
    assert df.groupby("object_id")["timestamp"].nunique().eq(1).all()
    assert set(df["object_id"]) == {s.object_id for s in TAGS.stream_signals("pipeline_segments")}
    assert df["parameter"].nunique() == 10


def test_cycle_scores_segments_with_caveats(model, ingested_db, tmp_path):
    db = tmp_path / "s.sqlite"
    shutil.copy(ingested_db, db)
    res = run_cycle(storage.connect(db), scc_model=model, hydrate_model=False, chudin_model=False, log=False, save=False)
    risks = {r.segment_id: r for r in res.scc_risks}
    assert len(risks) == 7
    assert risks["MG-1-034"].in_domain and not risks["PL-01-001"].in_domain
    assert all(r.synthetic_model and r.applicability == "unverified" for r in risks.values())
    for e in [e for e in res.events if e.code == "П17"]:
        assert e.message.startswith("ПРЕДВАРИТЕЛЬНО")
