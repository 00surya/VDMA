"""Small boundary checks; run the real checkpoint separately with smoke_model.py."""
from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from pydantic import ValidationError

import app as lab

PAYLOAD = {"state": "A parcel arrived damaged.", "question": "Which queue?", "options": ["shipping", "billing"]}
HEADERS = {"X-Decision-Client": "local-ui"}


class FakeModel:
    def __init__(self):
        self.processor = SimpleNamespace(
            tokenizer=SimpleNamespace(all_special_tokens=["[P]"]),
            _transform_record=lambda _: SimpleNamespace(input_ids=[1] * 32),
        )
        self.calls = 0

    def create_schema(self):
        return self

    def classification(self, *args, **kwargs):
        assert args == ("answer", ["shipping", "billing"])
        assert kwargs == {"prompt": "Which queue?", "multi_label": True, "class_act": "softmax", "cls_threshold": 0.0}
        return self

    def build(self):
        return {}

    def extract(self, *args, **kwargs):
        self.calls += 1
        return {"answer": [("billing", 0.2), ("shipping", 0.8)]}


@pytest.fixture
def engine():
    value = lab.DecisionEngine()
    value.status = "ready"
    value.model = FakeModel()
    return value


@pytest.fixture
def client(monkeypatch, engine):
    monkeypatch.setattr(lab, "engine", engine)
    # No lifespan: unit checks must not start the 1.95 GB checkpoint.
    return TestClient(lab.app, base_url="http://127.0.0.1:8771")


def test_distribution_and_single_call(engine):
    result = engine.decide(lab.DecisionRequest(**PAYLOAD))
    assert result["answer"] == "shipping"
    assert result["scores"] == [{"label": "shipping", "score": 0.8}, {"label": "billing", "score": 0.2}]
    assert result["margin"] == pytest.approx(0.6)
    assert result["calibrated"] is False
    assert engine.model.calls == 1


@pytest.mark.parametrize("changes", [{"state": " "}, {"question": " "}, {"options": ["same", " SAME "]}, {"options": ["one"]}, {"options": ["a", ""]}, {"options": ["a", 2]}, {"options": [str(i) for i in range(13)]}])
def test_invalid_input(changes):
    with pytest.raises(ValidationError):
        lab.DecisionRequest(**(PAYLOAD | changes))


def test_busy_and_loading(engine):
    request = lab.DecisionRequest(**PAYLOAD)
    engine.status = "loading"
    with pytest.raises(HTTPException) as err:
        engine.decide(request)
    assert err.value.status_code == 503
    engine.status = "ready"
    with engine.lock:
        with pytest.raises(HTTPException) as err:
            engine.decide(request)
    assert err.value.status_code == 429


def test_token_limit_reserved_and_lock_recovery(engine):
    with pytest.raises(HTTPException) as err:
        engine.decide(lab.DecisionRequest(**(PAYLOAD | {"state": "[P]"})))
    assert err.value.status_code == 422
    engine.model.processor._transform_record = lambda _: SimpleNamespace(input_ids=[1] * 513)
    with pytest.raises(HTTPException) as err:
        engine.decide(lab.DecisionRequest(**PAYLOAD))
    assert err.value.status_code == 422
    assert "nothing was truncated" in err.value.detail
    assert not engine.lock.locked()
    assert engine.model.calls == 0


def test_corrupt_scores_fail_closed(engine):
    for pairs in [[("shipping", float("nan")), ("billing", 0.2)], [("shipping", 0.8)], [("shipping", 0.8), ("billing", 0.8)]]:
        engine.model.extract = lambda *args, **kwargs: {"answer": pairs}
        with pytest.raises(HTTPException) as err:
            engine.decide(lab.DecisionRequest(**PAYLOAD))
        assert err.value.status_code == 500
        assert not engine.lock.locked()


def test_api_boundary_and_pages(client):
    assert client.post('/api/decide', json=PAYLOAD).status_code == 403
    assert client.post('/api/decide', json=PAYLOAD, headers=HEADERS | {"Origin": "https://other.example"}).status_code == 403
    assert client.post('/api/decide', json=PAYLOAD, headers=HEADERS | {"Origin": "null"}).status_code == 403
    assert client.get('/api/status', headers={"Host": "attacker.example"}).status_code == 400
    response = client.post('/api/decide', json=PAYLOAD, headers=HEADERS | {"Origin": "http://127.0.0.1:8771"})
    assert response.status_code == 200
    assert response.json()["answer"] == "shipping"
    assert response.headers["cache-control"] == "no-store"
    for page in ["/", "/explainer", "/static/app.js", "/static/style.css"]:
        assert client.get(page).status_code == 200
    assert client.get('/models/GLiNER2.5-Decide/model.safetensors').status_code == 404
    assert client.post('/api/decide', content='x'*33000, headers=HEADERS | {"Content-Type": "application/json"}).status_code == 413
    assert client.post('/api/decide', content='{}', headers=HEADERS | {"Content-Type": "text/plain"}).status_code == 415
    assert client.post('/api/decide', json=PAYLOAD | {"options": ["SECRET", "SECRET"]}, headers=HEADERS).status_code == 422
    assert "SECRET" not in client.post('/api/decide', json=PAYLOAD | {"unexpected": "SECRET"}, headers=HEADERS).text
