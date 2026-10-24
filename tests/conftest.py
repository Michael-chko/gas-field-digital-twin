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
