"""Этап 10: после установки зависимостей всё работает с отключённой сетью."""

import socket

import pytest
from fastapi.testclient import TestClient

from field_twin import config, hydrates, quality, storage
from field_twin.api import create_app
from field_twin.chudin_models import daily_table, train as train_chudin
from field_twin.cycle import CycleRunner
from field_twin.ingest import Ingestor
from field_twin.simulator import chudin_daily, generate

LOCAL = ("127.0.0.1", "localhost", "::1")


@pytest.fixture
def no_network(monkeypatch):
    """Запрет любых сетевых соединений, кроме локальных."""
    real = socket.socket.connect

    def guarded(self, address):
        host = address[0] if isinstance(address, tuple) else address
        if host not in LOCAL:
            raise OSError(f"сеть отключена: попытка соединения с {host}")
        return real(self, address)

    monkeypatch.setattr(socket.socket, "connect", guarded)   # create_connection тоже идёт через connect


def test_full_chain_without_network(no_network, tmp_path):
    cfg = config.simulator()
    cfg["defects"]["hydrates"]["episodes"] = 2
    res = generate(tmp_path / "s", days=3, seed=4, defects=cfg["defects"])
    conn = storage.connect(tmp_path / "t.sqlite")
    ing = Ingestor(conn)
    for p in res.files.values():
        ing.ingest_file(p)
    data = quality.QualityPipeline().run(quality.load_measurements(conn)).data
    s = hydrates.settings()
    feats = hydrates.build_features(data, s)
    hm = hydrates.train(feats, hydrates.labels_from_periods(feats, hydrates.periods_from_truth(res.truth)), s, trials=2)
    d = chudin_daily(120, seed=2)
    cm = train_chudin(daily_table(d, "dks"), daily_table(d, "cluster"), config.load_yaml("models.yaml")["chudin"])
    runner = CycleRunner(conn, hydrate_model=hm, chudin_model=cm)
    out = runner.step()
    assert out.hydrate_predictions is not None and out.model_errors
    st = TestClient(create_app(tmp_path / "t.sqlite")).get("/api/state", params={"role": "operator"}).json()
    assert st["snapshot"]["freshness"]


def test_guard_really_blocks(no_network):
    with pytest.raises(OSError, match="сеть отключена"):
        s = socket.socket()
        try:
            s.connect(("93.184.216.34", 80))
        finally:
            s.close()
