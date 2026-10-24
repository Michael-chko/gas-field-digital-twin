from datetime import timedelta

import pytest

from field_twin import config
from field_twin.gateway.tag_dictionary import TagDictionary


def test_thresholds_have_all_rules():
    th = config.thresholds()
    rules = {k: v for k, v in th.items() if k.startswith("П")}
    assert [f"П{i}" for i in range(1, 17)] == list(rules)
    assert th["configurable_recipient"] in ("operator", "chief_engineer")
    for code, rule in rules.items():
        assert rule.get("source"), f"у {code} нет источника"
        assert rule.get("recipient")


def test_source_thresholds_values():
    th = config.thresholds()
    assert (th["П1"]["min_positive"], th["П1"]["window_minutes"]) == (3, 20)
    assert th["П3"]["max_mm_per_year"] == 15
    assert th["П5"]["max_m_per_s"] == 20
    assert th["П13"]["stale_after_intervals"] == 2
    # Порогов нет в источниках — остаются пустыми до решения пользователя.
    assert th["П14"]["min_pearson"] is None
    assert th["П16"]["max_c5plus_g_m3"] is None


@pytest.mark.parametrize("text,expected", [
    ("2min", timedelta(minutes=2)), ("5min", timedelta(minutes=5)),
    ("30min", timedelta(minutes=30)), ("1h", timedelta(hours=1)), ("1d", timedelta(days=1)), (None, None),
])
def test_parse_interval(text, expected):
    assert config.parse_interval(text) == expected


def test_parse_interval_rejects_garbage():
    with pytest.raises(ValueError):
        config.parse_interval("пять минут")


def test_tag_dictionary_absorber_parameters():
    tags = TagDictionary.load()
    absorber = {s.parameter for s in tags.stream_signals("absorber")}
    # Обозначения как в таблице 2 статьи Сафроновой.
    for p in ["Q_gas", "KRU", "L_cube", "T_gas_absorber", "T_cond_irrigation", "T_heat_exchanger_1",
              "T_heat_exchanger_2", "T_heat_exchanger_3", "L_E101.1", "L_E101.2", "Q_cond", "Q_sikgk"]:
        assert p in absorber
    assert tags.stream_interval("absorber") == timedelta(minutes=2)
    assert tags.stream_interval("gathering") == timedelta(minutes=5)
    assert tags.stream_interval("cluster") == timedelta(days=1)
    assert tags.stream_interval("wells") is None


def test_every_signal_bound_to_object_and_tag():
    tags = TagDictionary.load()
    assert len(tags) > 50
    for sig in tags.signals.values():
        assert sig.object_id and sig.unit and sig.tag == f"{sig.stream}.{sig.object_id}.{sig.parameter}"
    ecp = tags.get("ecp", "ECP-D1", "current")
    assert ecp.object_attrs["install_type"] == "drainage"
