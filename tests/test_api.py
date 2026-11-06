import pytest
from fastapi.testclient import TestClient

from readmit.pipeline import run


@pytest.fixture(scope="module")
def client(tmp_path_factory, monkeypatch_module):
    out = tmp_path_factory.mktemp("art")
    run(None, out, synthetic=True, n_boot=20)
    monkeypatch_module.setenv("MODEL_DIR", str(out / "model"))
    from readmit import api
    api.load.cache_clear()
    return TestClient(api.app)


@pytest.fixture(scope="module")
def monkeypatch_module():
    mp = pytest.MonkeyPatch()
    yield mp
    mp.undo()


ENC = {"encounter_id": 1, "age": "[70-80)", "time_in_hospital": 5, "number_inpatient": 3,
       "diag_1": "428", "medications": {"insulin": "Up"}}


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"


def test_score_returns_probabilities(client):
    low = dict(ENC, encounter_id=2, number_inpatient=0)
    r = client.post("/score", json={"encounters": [ENC, low]})
    assert r.status_code == 200
    res = r.json()["results"]
    assert all(0 <= x["risk_30d"] <= 1 for x in res)


def test_rejects_invalid_length_of_stay(client):
    r = client.post("/score", json={"encounters": [dict(ENC, time_in_hospital=40)]})
    assert r.status_code == 422


def test_rejects_unknown_medication(client):
    r = client.post("/score", json={"encounters": [dict(ENC, medications={"aspirin": "Up"})]})
    assert r.status_code == 422
