import json

import pytest

from field_twin import storage
from field_twin.simulator import generate


@pytest.fixture(scope="session")
def samples(tmp_path_factory):
    """Синтетические данные за 2 суток с дефектами (один раз на сессию тестов)."""
    out = tmp_path_factory.mktemp("samples")
    res = generate(out, days=2, seed=7)
    return res


@pytest.fixture(scope="session")
def truth(samples):
    with open(samples.out_dir / "truth.json", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture
def conn():
    c = storage.connect(":memory:")
    yield c
    c.close()


@pytest.fixture(scope="session")
def ingested_db(samples, tmp_path_factory):
    """База с принятыми синтетическими данными (М1)."""
    from field_twin.ingest import Ingestor

    path = tmp_path_factory.mktemp("db") / "twin.sqlite"
    c = storage.connect(path)
    ing = Ingestor(c)
    for p in samples.files.values():
        ing.ingest_file(p)
    c.close()
    return path


@pytest.fixture(scope="session")
def cleaned(ingested_db):
    """Результат М2 на синтетических данных с настройками из config/."""
    from field_twin import quality

    c = storage.connect(ingested_db)
    res = quality.QualityPipeline().run(quality.load_measurements(c))
    c.close()
    return res
